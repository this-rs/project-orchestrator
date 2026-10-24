//! HMAC-only crypto provider for `jsonwebtoken`.
//!
//! Every JWT this backend signs or verifies is HS256 (session, vault, MCP, OAuth client id and
//! consent tokens: see `jwt.rs`, `oauth_server.rs`). The crate's `rust_crypto` feature compiles
//! every algorithm instead (ECDSA, RSA, Ed25519) and drags in `elliptic-curve` and `crypto-bigint
//! 0.5.5` (RUSTSEC-2026-0387) and `rsa` (RUSTSEC-2023-0071) for code that never runs here.
//!
//! This provider serves the HMAC family only, with the same `hmac`/`sha2` crates and the same
//! logic as the crate's own HMAC provider. Any other algorithm is refused with
//! `InvalidAlgorithm`, so a token claiming `RS256` or `ES256` is rejected at the provider, not
//! merely by the `Validation` list.
//!
//! `jsonwebtoken` panics if no provider is installed, so the backend calls [`encode`] and
//! [`decode`] of this module (which install it first) rather than the crate's own.

use std::sync::Once;

use hmac::{Hmac, KeyInit, Mac};
use jsonwebtoken::crypto::{CryptoProvider, JwtSigner, JwtVerifier, KeyUtils};
use jsonwebtoken::errors::{ErrorKind, Result};
use jsonwebtoken::signature::{Signer, Verifier};
use jsonwebtoken::{
    Algorithm, AlgorithmFamily, DecodingKey, EncodingKey, Header, TokenData, Validation,
};
use serde::de::DeserializeOwned;
use serde::Serialize;
use sha2::{Sha256, Sha384, Sha512};

macro_rules! hmac_signer {
    ($name:ident, $alg:expr, $hmac:ty) => {
        struct $name($hmac);

        impl $name {
            fn new(key: &EncodingKey) -> Result<Self> {
                if key.family() != AlgorithmFamily::Hmac {
                    return Err(ErrorKind::InvalidKeyFormat.into());
                }
                <$hmac>::new_from_slice(key.as_bytes())
                    .map(Self)
                    .map_err(|_| ErrorKind::InvalidKeyFormat.into())
            }
        }

        impl Signer<Vec<u8>> for $name {
            fn try_sign(
                &self,
                msg: &[u8],
            ) -> std::result::Result<Vec<u8>, jsonwebtoken::signature::Error> {
                let mut mac = self.0.clone();
                mac.update(msg);
                Ok(mac.finalize().into_bytes().to_vec())
            }
        }

        impl JwtSigner for $name {
            fn algorithm(&self) -> Algorithm {
                $alg
            }
        }
    };
}

macro_rules! hmac_verifier {
    ($name:ident, $alg:expr, $hmac:ty) => {
        struct $name($hmac);

        impl $name {
            fn new(key: &DecodingKey) -> Result<Self> {
                if key.family() != AlgorithmFamily::Hmac {
                    return Err(ErrorKind::InvalidKeyFormat.into());
                }
                <$hmac>::new_from_slice(key.try_get_as_bytes()?)
                    .map(Self)
                    .map_err(|_| ErrorKind::InvalidKeyFormat.into())
            }
        }

        impl Verifier<Vec<u8>> for $name {
            fn verify(
                &self,
                msg: &[u8],
                signature: &Vec<u8>,
            ) -> std::result::Result<(), jsonwebtoken::signature::Error> {
                let mut mac = self.0.clone();
                mac.update(msg);
                mac.verify_slice(signature)
                    .map_err(jsonwebtoken::signature::Error::from_source)
            }
        }

        impl JwtVerifier for $name {
            fn algorithm(&self) -> Algorithm {
                $alg
            }
        }
    };
}

hmac_signer!(Hs256Signer, Algorithm::HS256, Hmac<Sha256>);
hmac_signer!(Hs384Signer, Algorithm::HS384, Hmac<Sha384>);
hmac_signer!(Hs512Signer, Algorithm::HS512, Hmac<Sha512>);
hmac_verifier!(Hs256Verifier, Algorithm::HS256, Hmac<Sha256>);
hmac_verifier!(Hs384Verifier, Algorithm::HS384, Hmac<Sha384>);
hmac_verifier!(Hs512Verifier, Algorithm::HS512, Hmac<Sha512>);

fn signer(alg: &Algorithm, key: &EncodingKey) -> Result<Box<dyn JwtSigner>> {
    match alg {
        Algorithm::HS256 => Ok(Box::new(Hs256Signer::new(key)?)),
        Algorithm::HS384 => Ok(Box::new(Hs384Signer::new(key)?)),
        Algorithm::HS512 => Ok(Box::new(Hs512Signer::new(key)?)),
        _ => Err(ErrorKind::InvalidAlgorithm.into()),
    }
}

fn verifier(alg: &Algorithm, key: &DecodingKey) -> Result<Box<dyn JwtVerifier>> {
    match alg {
        Algorithm::HS256 => Ok(Box::new(Hs256Verifier::new(key)?)),
        Algorithm::HS384 => Ok(Box::new(Hs384Verifier::new(key)?)),
        Algorithm::HS512 => Ok(Box::new(Hs512Verifier::new(key)?)),
        _ => Err(ErrorKind::InvalidAlgorithm.into()),
    }
}

static HMAC_ONLY: CryptoProvider = CryptoProvider {
    signer_factory: signer,
    verifier_factory: verifier,
    // JWK/PEM key conversion is never used here (HMAC keys are raw secrets).
    key_utils: KeyUtils::new_unimplemented(),
};

/// Install the HMAC-only provider for this process (idempotent, safe from any thread).
pub fn ensure_installed() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // `Err` means a provider was already installed: nothing to do.
        let _ = HMAC_ONLY.install_default();
    });
}

/// `jsonwebtoken::encode`, with the provider installed first.
pub fn encode<T: Serialize>(header: &Header, claims: &T, key: &EncodingKey) -> Result<String> {
    ensure_installed();
    jsonwebtoken::encode(header, claims, key)
}

/// `jsonwebtoken::decode`, with the provider installed first.
pub fn decode<T: DeserializeOwned>(
    token: impl AsRef<[u8]>,
    key: &DecodingKey,
    validation: &Validation,
) -> Result<TokenData<T>> {
    ensure_installed();
    jsonwebtoken::decode(token, key, validation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct C {
        sub: String,
        exp: u64,
    }

    fn claims() -> C {
        C {
            sub: "u".into(),
            exp: jsonwebtoken::get_current_timestamp() + 60,
        }
    }

    #[test]
    fn hs256_round_trips() {
        let key = EncodingKey::from_secret(b"0123456789abcdef0123456789abcdef");
        let token = encode(&Header::default(), &claims(), &key).unwrap();
        let back: TokenData<C> = decode(
            &token,
            &DecodingKey::from_secret(b"0123456789abcdef0123456789abcdef"),
            &Validation::new(Algorithm::HS256),
        )
        .unwrap();
        assert_eq!(back.claims.sub, "u");
    }

    #[test]
    fn a_wrong_secret_is_rejected() {
        let token = encode(
            &Header::default(),
            &claims(),
            &EncodingKey::from_secret(b"0123456789abcdef0123456789abcdef"),
        )
        .unwrap();
        let err = decode::<C>(
            &token,
            &DecodingKey::from_secret(b"fedcba9876543210fedcba9876543210"),
            &Validation::new(Algorithm::HS256),
        )
        .unwrap_err();
        assert!(matches!(err.kind(), ErrorKind::InvalidSignature), "{err:?}");
    }

    #[test]
    fn hs384_and_hs512_work_too() {
        for alg in [Algorithm::HS384, Algorithm::HS512] {
            let key = EncodingKey::from_secret(b"0123456789abcdef0123456789abcdef");
            let token = encode(&Header::new(alg), &claims(), &key).unwrap();
            decode::<C>(
                &token,
                &DecodingKey::from_secret(b"0123456789abcdef0123456789abcdef"),
                &Validation::new(alg),
            )
            .unwrap();
        }
    }

    #[test]
    fn a_non_hmac_algorithm_is_refused_by_the_provider() {
        ensure_installed();
        let key = EncodingKey::from_secret(b"0123456789abcdef0123456789abcdef");
        for alg in [Algorithm::RS256, Algorithm::ES256, Algorithm::EdDSA] {
            assert!(
                signer(&alg, &key).is_err(),
                "{alg:?} must not be signable here"
            );
            assert!(
                verifier(&alg, &DecodingKey::from_secret(b"k")).is_err(),
                "{alg:?} must not be verifiable here"
            );
        }
    }

    #[test]
    fn a_token_forged_as_none_or_rs256_is_rejected() {
        // header {"alg":"RS256","typ":"JWT"} . {"sub":"u","exp":9999999999} . junk
        let forged =
            "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1IiwiZXhwIjo5OTk5OTk5OTk5fQ.AAAA";
        let r = decode::<C>(
            forged,
            &DecodingKey::from_secret(b"0123456789abcdef0123456789abcdef"),
            &Validation::new(Algorithm::HS256),
        );
        assert!(r.is_err());
    }
}
