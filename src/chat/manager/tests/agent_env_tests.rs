    use super::server_secrets_to_hide;
    #[test]
    fn only_the_server_secrets_that_exist_are_hidden() {
        let present = |k: &str| k == "NEO4J_PASSWORD" || k == "PO_JWT_SECRET";
        assert_eq!(
            server_secrets_to_hide(present),
            vec!["NEO4J_PASSWORD", "PO_JWT_SECRET"]
        );
        assert!(server_secrets_to_hide(|_| false).is_empty());
    }

    #[test]
    fn the_cli_keeps_the_credentials_it_needs_to_authenticate() {
        let hidden = server_secrets_to_hide(|_| true);
        for needed in [
            "ANTHROPIC_API_KEY",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "PATH",
            "HOME",
        ] {
            assert!(
                !hidden.contains(&needed),
                "{needed} must stay visible to the CLI"
            );
        }
        for secret in [
            "NEO4J_PASSWORD",
            "MEILISEARCH_KEY",
            "EMBEDDING_API_KEY",
            "PO_JWT_SECRET",
        ] {
            assert!(
                hidden.contains(&secret),
                "{secret} must be hidden from agents"
            );
        }
    }
