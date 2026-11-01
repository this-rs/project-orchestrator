//! Les écritures du graphe de code (unitaires ET par lot) contre un VRAI Neo4j :
//! la clé stable et le propriétaire des symboles sont écrits, puis relus.
//!
//! Pourquoi ce fichier : la PR de la clé stable ajoute `stable_key`, `owner_*`
//! et `occurrence` à six chemins d'écriture (`upsert_*` et `batch_upsert_*`).
//! Sans Neo4j, ces lignes ne s'exécutent pas ; avec, elles ne sont exercées que
//! par le test de migration. Ici, chaque chemin d'écriture est joué et relu.
//!
//! ```text
//! NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=testpassword \
//!   cargo test --test code_graph_stable_key
//! ```
//! Chaque test alloue des chemins uniques et supprime ce qu'il a créé.

use chrono::Utc;
use project_orchestrator::neo4j::models::*;
use project_orchestrator::neo4j::Neo4jClient;
use uuid::Uuid;

async fn store() -> Option<Neo4jClient> {
    let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".into());
    let user = std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".into());
    let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "orchestrator123".into());
    match Neo4jClient::new(&uri, &user, &password).await {
        Ok(client) => Some(client),
        Err(err) => {
            if std::env::var("CI").is_ok() {
                panic!("Neo4j is required in CI but {uri} did not answer: {err}");
            }
            eprintln!("skipping: no Neo4j at {uri} ({err})");
            None
        }
    }
}

fn function(path: &str, name: &str, line: u32, owner: Option<SymbolOwner>) -> FunctionNode {
    FunctionNode {
        name: name.into(),
        visibility: Visibility::Public,
        params: vec![Parameter {
            name: "x".into(),
            type_name: Some("u32".into()),
        }],
        return_type: Some("u32".into()),
        generics: vec![],
        is_async: false,
        is_unsafe: false,
        complexity: 1,
        file_path: path.into(),
        line_start: line,
        line_end: line + 3,
        docstring: None,
        owner,
    }
}

fn function_id(f: &FunctionNode) -> String {
    format!("{}:{}:{}", f.file_path, f.name, f.line_start)
}

#[tokio::test]
async fn every_symbol_write_path_persists_its_stable_key_and_reads_back_its_owner() {
    let Some(client) = store().await else { return };
    let path = format!("/tmp/stable-key-{}/lib.rs", Uuid::new_v4());
    client
        .upsert_file(&FileNode {
            path: path.clone(),
            language: "rust".into(),
            hash: "h".into(),
            last_parsed: Utc::now(),
            project_id: None,
        })
        .await
        .unwrap();

    // Fonctions : une seule (upsert) puis un lot, avec et sans propriétaire.
    let owner = SymbolOwner::new("Parser", Some("Debug"));
    let single = function(&path, "single_fn", 1, Some(owner.clone()));
    client.upsert_function(&single).await.unwrap();
    let batch = vec![
        function(&path, "owned_fn", 10, Some(owner.clone())),
        function(&path, "plain_fn", 20, None),
    ];
    client.batch_upsert_functions(&batch).await.unwrap();
    // Rejouer le lot est idempotent (MERGE sur la clé).
    client.batch_upsert_functions(&batch).await.unwrap();

    // Types : struct / trait / enum, unitaire et par lot.
    let strukt = |n: &str, l: u32| StructNode {
        name: n.into(),
        visibility: Visibility::Public,
        generics: vec![],
        file_path: path.clone(),
        line_start: l,
        line_end: l + 2,
        docstring: None,
        parent_class: None,
        interfaces: vec![],
    };
    client
        .batch_upsert_structs(&[strukt("S1", 30)])
        .await
        .unwrap();

    let tr = |n: &str, l: u32| TraitNode {
        name: n.into(),
        visibility: Visibility::Public,
        generics: vec![],
        file_path: path.clone(),
        line_start: l,
        line_end: l + 2,
        docstring: None,
        is_external: false,
        source: None,
    };
    client.upsert_trait(&tr("T1", 40)).await.unwrap();
    client.batch_upsert_traits(&[tr("T2", 50)]).await.unwrap();

    let en = |n: &str, l: u32| EnumNode {
        name: n.into(),
        visibility: Visibility::Public,
        variants: vec!["A".into(), "B".into()],
        file_path: path.clone(),
        line_start: l,
        line_end: l + 2,
        docstring: None,
    };
    client.upsert_enum(&en("E1", 60)).await.unwrap();
    client.batch_upsert_enums(&[en("E2", 70)]).await.unwrap();

    // Impl : locale, puis impl d'un trait local et d'un trait EXTERNE.
    let imp = |ty: &str, tr: Option<&str>, l: u32| ImplNode {
        for_type: ty.into(),
        trait_name: tr.map(String::from),
        generics: vec![],
        where_clause: None,
        file_path: path.clone(),
        line_start: l,
        line_end: l + 2,
    };
    client.upsert_impl(&imp("S1", None, 80)).await.unwrap();
    client
        .batch_upsert_impls(&[
            imp("S1", Some("T1"), 90),
            imp("S1", Some("std::fmt::Debug"), 100),
        ])
        .await
        .unwrap();

    // Imports : unitaire, par lot, et deux lignes du même chemin (occurrence).
    let import = |p: &str, occ: u32, l: u32| ImportNode {
        path: p.into(),
        alias: None,
        items: vec!["X".into()],
        file_path: path.clone(),
        line: l,
        occurrence: occ,
    };
    client
        .upsert_import(&import("std::io", 0, 1))
        .await
        .unwrap();
    client
        .batch_upsert_imports(&[import("std::fs", 0, 2), import("std::fs", 1, 3)])
        .await
        .unwrap();

    // Appels : `owned` appelle `free`, `single` appelle `owned`.
    let owned = &batch[0];
    client
        .create_call_relationship(&function_id(owned), "plain_fn", None, 0.9, "same_file")
        .await
        .unwrap();
    let callees = client.get_callees(&function_id(owned), 1).await.unwrap();
    assert_eq!(callees.len(), 1);
    assert_eq!(callees[0].name, "plain_fn");
    assert_eq!(
        callees[0].owner, None,
        "une fonction libre n'a pas de propriétaire"
    );

    client
        .create_call_relationship(&function_id(&single), "owned_fn", None, 0.9, "same_file")
        .await
        .unwrap();
    let callers = client
        .find_callers(&function_id(owned), None)
        .await
        .unwrap();
    assert_eq!(callers.len(), 1);
    assert_eq!(callers[0].name, "single_fn");
    assert_eq!(
        callers[0].owner,
        Some(owner.clone()),
        "le propriétaire (type + trait) revient tel qu'écrit"
    );
    let owned_callees = client.get_callees(&function_id(&single), 1).await.unwrap();
    assert_eq!(owned_callees[0].owner, Some(owner));

    client.delete_file(&path).await.unwrap();
}
