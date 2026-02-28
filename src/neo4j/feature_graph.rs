//! Neo4j Feature Graph operations

use super::client::Neo4jClient;
use super::models::*;
use anyhow::Result;
use neo4rs::query;
use uuid::Uuid;

impl Neo4jClient {
    // ================================================================
    // Feature Graphs
    // ================================================================

    pub async fn create_feature_graph(&self, fg: &FeatureGraphNode) -> Result<()> {
        let q = query(
            "CREATE (fg:FeatureGraph {
                id: $id,
                name: $name,
                description: $description,
                project_id: $project_id,
                created_at: $created_at,
                updated_at: $updated_at,
                entry_function: $entry_function,
                build_depth: $build_depth,
                include_relations: $include_relations
            })
            WITH fg
            MATCH (p:Project {id: $project_id})
            CREATE (fg)-[:BELONGS_TO]->(p)",
        )
        .param("id", fg.id.to_string())
        .param("name", fg.name.clone())
        .param("description", fg.description.clone().unwrap_or_default())
        .param("project_id", fg.project_id.to_string())
        .param("created_at", fg.created_at.to_rfc3339())
        .param("updated_at", fg.updated_at.to_rfc3339())
        .param(
            "entry_function",
            fg.entry_function.clone().unwrap_or_default(),
        )
        .param("build_depth", fg.build_depth.map(|d| d as i64).unwrap_or(0))
        .param(
            "include_relations",
            fg.include_relations
                .as_ref()
                .map(|r| serde_json::to_string(r).unwrap_or_default())
                .unwrap_or_default(),
        );

        self.execute_with_params(q).await?;
        Ok(())
    }

    pub async fn get_feature_graph(&self, id: Uuid) -> Result<Option<FeatureGraphNode>> {
        let q = query("MATCH (fg:FeatureGraph {id: $id}) RETURN fg").param("id", id.to_string());

        let rows = self.execute_with_params(q).await?;
        if let Some(row) = rows.first() {
            let node: neo4rs::Node = row.get("fg")?;
            Ok(Some(self.node_to_feature_graph(&node)?))
        } else {
            Ok(None)
        }
    }

    pub async fn get_feature_graph_detail(&self, id: Uuid) -> Result<Option<FeatureGraphDetail>> {
        let fg = self.get_feature_graph(id).await?;
        let Some(fg) = fg else { return Ok(None) };

        let q = query(
            "MATCH (fg:FeatureGraph {id: $id})-[r:INCLUDES_ENTITY]->(e)
             RETURN labels(e)[0] AS entity_type,
                    COALESCE(e.path, e.id) AS entity_id,
                    COALESCE(e.name, e.path) AS name,
                    r.role AS role,
                    e.pagerank AS pagerank",
        )
        .param("id", id.to_string());

        let rows = self.execute_with_params(q).await?;

        // Find max pagerank for normalization (0.0–1.0 within the feature graph)
        let max_pr = rows
            .iter()
            .filter_map(|row| row.get::<f64>("pagerank").ok())
            .fold(0.0_f64, f64::max);

        let entities = rows
            .iter()
            .map(|row| {
                let raw_pr: Option<f64> = row.get::<f64>("pagerank").ok();
                let importance_score = if max_pr > 0.0 {
                    raw_pr.map(|pr| pr / max_pr)
                } else {
                    None
                };
                FeatureGraphEntity {
                    entity_type: row.get::<String>("entity_type").unwrap_or_default(),
                    entity_id: row.get::<String>("entity_id").unwrap_or_default(),
                    name: row.get::<String>("name").ok(),
                    role: row.get::<String>("role").ok(),
                    importance_score,
                }
            })
            .collect();

        // Fetch intra-graph relations (CALLS, IMPORTS, EXTENDS, IMPLEMENTS etc.)
        let rel_q = query(
            "MATCH (fg:FeatureGraph {id: $id})-[:INCLUDES_ENTITY]->(a)
             MATCH (fg)-[:INCLUDES_ENTITY]->(b)
             WHERE a <> b
             MATCH (a)-[r:CALLS|IMPORTS|EXTENDS|IMPLEMENTS|IMPLEMENTS_TRAIT|IMPLEMENTS_FOR]->(b)
             RETURN labels(a)[0] AS source_type,
                    COALESCE(a.path, a.name, a.id) AS source_id,
                    labels(b)[0] AS target_type,
                    COALESCE(b.path, b.name, b.id) AS target_id,
                    type(r) AS relation_type",
        )
        .param("id", id.to_string());

        let rel_rows = self.execute_with_params(rel_q).await?;
        let relations = rel_rows
            .iter()
            .map(|row| FeatureGraphRelation {
                source_type: row.get::<String>("source_type").unwrap_or_default(),
                source_id: row.get::<String>("source_id").unwrap_or_default(),
                target_type: row.get::<String>("target_type").unwrap_or_default(),
                target_id: row.get::<String>("target_id").unwrap_or_default(),
                relation_type: row.get::<String>("relation_type").unwrap_or_default(),
            })
            .collect();

        Ok(Some(FeatureGraphDetail {
            graph: fg,
            entities,
            relations,
        }))
    }

    pub async fn list_feature_graphs(
        &self,
        project_id: Option<Uuid>,
    ) -> Result<Vec<FeatureGraphNode>> {
        let cypher = if project_id.is_some() {
            "MATCH (fg:FeatureGraph {project_id: $pid})
             OPTIONAL MATCH (fg)-[:INCLUDES_ENTITY]->(e)
             RETURN fg, count(e) AS entity_count
             ORDER BY fg.updated_at DESC"
        } else {
            "MATCH (fg:FeatureGraph)
             OPTIONAL MATCH (fg)-[:INCLUDES_ENTITY]->(e)
             RETURN fg, count(e) AS entity_count
             ORDER BY fg.updated_at DESC"
        };

        let q = query(cypher).param(
            "pid",
            project_id.map(|id| id.to_string()).unwrap_or_default(),
        );

        let rows = self.execute_with_params(q).await?;
        let mut graphs = Vec::new();
        for row in &rows {
            let node: neo4rs::Node = row.get("fg")?;
            let mut fg = self.node_to_feature_graph(&node)?;
            fg.entity_count = row.get::<i64>("entity_count").ok();
            graphs.push(fg);
        }
        Ok(graphs)
    }

    pub async fn delete_feature_graph(&self, id: Uuid) -> Result<bool> {
        let q = query(
            "MATCH (fg:FeatureGraph {id: $id})
             DETACH DELETE fg
             RETURN count(fg) AS deleted",
        )
        .param("id", id.to_string());

        let rows = self.execute_with_params(q).await?;
        if let Some(row) = rows.first() {
            let deleted: i64 = row.get("deleted").unwrap_or(0);
            Ok(deleted > 0)
        } else {
            Ok(false)
        }
    }

    pub async fn add_entity_to_feature_graph(
        &self,
        feature_graph_id: Uuid,
        entity_type: &str,
        entity_id: &str,
        role: Option<&str>,
        project_id: Option<Uuid>,
    ) -> Result<()> {
        let role_clause = if role.is_some() {
            " SET r.role = $role"
        } else {
            ""
        };

        // When project_id is provided, scope Function/Struct/Trait/Enum matches
        // to prevent cross-project contamination. File uses unique `path`, so no scoping needed.
        let has_project = project_id.is_some();

        let cypher = match entity_type.to_lowercase().as_str() {
            "file" => format!(
                "MATCH (fg:FeatureGraph {{id: $fg_id}})
                 MATCH (e:File {{path: $entity_id}})
                 MERGE (fg)-[r:INCLUDES_ENTITY]->(e){}",
                role_clause
            ),
            "function" if has_project => format!(
                "MATCH (fg:FeatureGraph {{id: $fg_id}})
                 MATCH (e:Function {{name: $entity_id}})<-[:CONTAINS]-(:File)<-[:CONTAINS]-(:Project {{id: $project_id}})
                 MERGE (fg)-[r:INCLUDES_ENTITY]->(e){}",
                role_clause
            ),
            "function" => format!(
                "MATCH (fg:FeatureGraph {{id: $fg_id}})
                 MATCH (e:Function {{name: $entity_id}})
                 MERGE (fg)-[r:INCLUDES_ENTITY]->(e){}",
                role_clause
            ),
            "struct" if has_project => format!(
                "MATCH (fg:FeatureGraph {{id: $fg_id}})
                 MATCH (e:Struct {{name: $entity_id}})<-[:CONTAINS]-(:File)<-[:CONTAINS]-(:Project {{id: $project_id}})
                 MERGE (fg)-[r:INCLUDES_ENTITY]->(e){}",
                role_clause
            ),
            "struct" => format!(
                "MATCH (fg:FeatureGraph {{id: $fg_id}})
                 MATCH (e:Struct {{name: $entity_id}})
                 MERGE (fg)-[r:INCLUDES_ENTITY]->(e){}",
                role_clause
            ),
            "trait" if has_project => format!(
                "MATCH (fg:FeatureGraph {{id: $fg_id}})
                 MATCH (e:Trait {{name: $entity_id}})<-[:CONTAINS]-(:File)<-[:CONTAINS]-(:Project {{id: $project_id}})
                 MERGE (fg)-[r:INCLUDES_ENTITY]->(e){}",
                role_clause
            ),
            "trait" => format!(
                "MATCH (fg:FeatureGraph {{id: $fg_id}})
                 MATCH (e:Trait {{name: $entity_id}})
                 MERGE (fg)-[r:INCLUDES_ENTITY]->(e){}",
                role_clause
            ),
            "enum" if has_project => format!(
                "MATCH (fg:FeatureGraph {{id: $fg_id}})
                 MATCH (e:Enum {{name: $entity_id}})<-[:CONTAINS]-(:File)<-[:CONTAINS]-(:Project {{id: $project_id}})
                 MERGE (fg)-[r:INCLUDES_ENTITY]->(e){}",
                role_clause
            ),
            "enum" => format!(
                "MATCH (fg:FeatureGraph {{id: $fg_id}})
                 MATCH (e:Enum {{name: $entity_id}})
                 MERGE (fg)-[r:INCLUDES_ENTITY]->(e){}",
                role_clause
            ),
            _ => {
                return Err(anyhow::anyhow!(
                    "Unsupported entity type for feature graph: {}",
                    entity_type
                ));
            }
        };

        let mut q = query(&cypher)
            .param("fg_id", feature_graph_id.to_string())
            .param("entity_id", entity_id.to_string());
        if let Some(r) = role {
            q = q.param("role", r.to_string());
        }
        if let Some(pid) = project_id {
            q = q.param("project_id", pid.to_string());
        }
        self.execute_with_params(q).await?;

        let update_q = query("MATCH (fg:FeatureGraph {id: $id}) SET fg.updated_at = $now")
            .param("id", feature_graph_id.to_string())
            .param("now", chrono::Utc::now().to_rfc3339());
        self.execute_with_params(update_q).await?;

        Ok(())
    }

    pub async fn remove_entity_from_feature_graph(
        &self,
        feature_graph_id: Uuid,
        entity_type: &str,
        entity_id: &str,
    ) -> Result<bool> {
        let match_prop = match entity_type.to_lowercase().as_str() {
            "file" => "path",
            _ => "name",
        };
        let label = match entity_type.to_lowercase().as_str() {
            "file" => "File",
            "function" => "Function",
            "struct" => "Struct",
            "trait" => "Trait",
            "enum" => "Enum",
            _ => return Err(anyhow::anyhow!("Unsupported entity type: {}", entity_type)),
        };

        let cypher = format!(
            "MATCH (fg:FeatureGraph {{id: $fg_id}})-[r:INCLUDES_ENTITY]->(e:{} {{{}: $entity_id}})
             DELETE r
             RETURN count(r) AS deleted",
            label, match_prop
        );

        let q = query(&cypher)
            .param("fg_id", feature_graph_id.to_string())
            .param("entity_id", entity_id.to_string());

        let rows = self.execute_with_params(q).await?;
        if let Some(row) = rows.first() {
            let deleted: i64 = row.get("deleted").unwrap_or(0);
            Ok(deleted > 0)
        } else {
            Ok(false)
        }
    }

    /// Expand a feature graph by finding semantically similar functions via vector search.
    ///
    /// For each input function that has an embedding stored in Neo4j, performs HNSW vector
    /// search to find the top-5 most similar functions in the project. Results are filtered
    /// by cosine score > 0.8 and optionally by Louvain community membership.
    ///
    /// Returns `(function_name, file_path, cosine_score)` for new functions not already
    /// in the input set, deduplicated by keeping the best score per function.
    ///
    /// Best-effort: returns empty Vec on any error (embeddings not available, index missing, etc.)
    pub async fn expand_by_vector_similarity(
        &self,
        functions: &[String],
        project_id: Uuid,
        community_filter: Option<i64>,
    ) -> Vec<(String, String, f64)> {
        let func_set: std::collections::HashSet<&str> =
            functions.iter().map(|s| s.as_str()).collect();

        // Batch-retrieve embeddings for all input functions
        let q = query(
            r#"
            MATCH (fn:Function)<-[:CONTAINS]-(f:File)<-[:CONTAINS]-(p:Project {id: $project_id})
            WHERE fn.name IN $names AND fn.embedding IS NOT NULL
            RETURN fn.name AS name, fn.embedding AS embedding
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("names", functions.to_vec());

        let rows = match self.execute_with_params(q).await {
            Ok(rows) => rows,
            Err(e) => {
                tracing::debug!(
                    "expand_by_vector_similarity: failed to get embeddings: {}",
                    e
                );
                return Vec::new();
            }
        };

        if rows.is_empty() {
            return Vec::new();
        }

        // For each function with an embedding, do vector search for neighbors
        let mut candidates: Vec<(String, String, f64)> = Vec::new();
        for row in &rows {
            let _name: String = match row.get("name") {
                Ok(n) => n,
                Err(_) => continue,
            };
            let embedding: Vec<f64> = match row.get("embedding") {
                Ok(e) => e,
                Err(_) => continue,
            };
            let embedding_f32: Vec<f32> = embedding.iter().map(|&x| x as f32).collect();

            let neighbors = match self
                .vector_search_functions(&embedding_f32, 5, Some(project_id))
                .await
            {
                Ok(n) => n,
                Err(_) => continue,
            };

            for (func_name, file_path, score) in neighbors {
                if func_set.contains(func_name.as_str()) || score < 0.8 {
                    continue;
                }
                candidates.push((func_name, file_path, score));
            }
        }

        // Deduplicate: keep best score per function
        let mut best: std::collections::HashMap<String, (String, f64)> =
            std::collections::HashMap::new();
        for (name, path, score) in candidates {
            let entry = best.entry(name).or_insert((path.clone(), 0.0));
            if score > entry.1 {
                *entry = (path, score);
            }
        }

        // Apply community filter if provided — single batch query
        if let Some(target_cid) = community_filter {
            let candidate_names: Vec<String> = best.keys().cloned().collect();
            if !candidate_names.is_empty() {
                let cq = query(
                    r#"
                    MATCH (fn:Function)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p:Project {id: $project_id})
                    WHERE fn.name IN $names
                    RETURN fn.name AS name, fn.community_id AS cid
                    "#,
                )
                .param("project_id", project_id.to_string())
                .param("names", candidate_names);

                if let Ok(crows) = self.execute_with_params(cq).await {
                    let mut allowed: std::collections::HashSet<String> =
                        std::collections::HashSet::new();
                    for crow in &crows {
                        if let Ok(name) = crow.get::<String>("name") {
                            let cid: Option<i64> = crow.get::<i64>("cid").ok();
                            // Allow if same community or no community data
                            if cid.is_none() || cid == Some(target_cid) {
                                allowed.insert(name);
                            }
                        }
                    }
                    best.retain(|name, _| allowed.contains(name));
                }
            }
        }

        best.into_iter()
            .map(|(name, (path, score))| (name, path, score))
            .collect()
    }

    /// Expand a feature graph by finding related files from the same Louvain community.
    ///
    /// For each input file that has a `community_id` in Neo4j, finds other files in the
    /// same community sorted by PageRank (descending), limited to `max_per_community` per
    /// community. Only returns files not already in the input set.
    ///
    /// Uses a single Cypher query for efficiency (no N+1 pattern).
    ///
    /// Best-effort: returns empty Vec on any error (GDS not run, no community data, etc.)
    pub async fn expand_by_community(
        &self,
        file_paths: &[String],
        project_id: Uuid,
        max_per_community: usize,
    ) -> Vec<String> {
        if file_paths.is_empty() {
            return Vec::new();
        }

        // Single query: for each input file's community, find top-N peers by pagerank
        let q = query(
            r#"
            MATCH (source:File)<-[:CONTAINS]-(p:Project {id: $project_id})
            WHERE source.path IN $paths AND source.community_id IS NOT NULL
            WITH DISTINCT source.community_id AS cid, p
            MATCH (peer:File)<-[:CONTAINS]-(p)
            WHERE peer.community_id = cid AND NOT peer.path IN $paths
            WITH cid, peer
            ORDER BY COALESCE(peer.pagerank, 0.0) DESC
            WITH cid, collect(peer.path)[..$limit] AS top_peers
            UNWIND top_peers AS peer_path
            RETURN DISTINCT peer_path
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("paths", file_paths.to_vec())
        .param("limit", max_per_community as i64);

        match self.execute_with_params(q).await {
            Ok(rows) => rows
                .iter()
                .filter_map(|row| row.get::<String>("peer_path").ok())
                .collect(),
            Err(e) => {
                tracing::debug!("expand_by_community failed: {}", e);
                Vec::new()
            }
        }
    }

    /// Standard/derive traits that are noise in feature graphs.
    /// These are ubiquitous traits (std, serde, tokio) that don't convey feature-specific meaning.
    const FEATURE_GRAPH_TRAIT_BLACKLIST: &'static [&'static str] = &[
        // std derives
        "Debug",
        "Display",
        "Clone",
        "Copy",
        "Default",
        "PartialEq",
        "Eq",
        "PartialOrd",
        "Ord",
        "Hash",
        "From",
        "Into",
        "TryFrom",
        "TryInto",
        "AsRef",
        "AsMut",
        "Deref",
        "DerefMut",
        "Drop",
        "Send",
        "Sync",
        "Sized",
        // serde
        "Serialize",
        "Deserialize",
        "Serializer",
        "Deserializer",
        // tokio / async
        "AsyncRead",
        "AsyncWrite",
        "AsyncSeek",
        "AsyncBufRead",
        // common derives
        "Error",
        "Future",
        "Stream",
    ];

    /// Check if a trait name is in the standard blacklist
    fn is_blacklisted_trait(name: &str) -> bool {
        Self::FEATURE_GRAPH_TRAIT_BLACKLIST.contains(&name)
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn auto_build_feature_graph(
        &self,
        name: &str,
        description: Option<&str>,
        project_id: Uuid,
        entry_function: &str,
        depth: u32,
        include_relations: Option<&[String]>,
        filter_community: Option<bool>,
    ) -> Result<FeatureGraphDetail> {
        // Helper: check if a relation type should be included
        let should_include = |rel: &str| -> bool {
            match &include_relations {
                None => true, // default: include all
                Some(rels) => rels.iter().any(|r| r.eq_ignore_ascii_case(rel)),
            }
        };

        let filter_community = filter_community.unwrap_or(true);

        // Step 1: Collect all related functions and their files via call graph
        // We collect the entry function + callers + callees, plus community_id for filtering
        let depth = depth.clamp(1, 5);

        let q = query(&format!(
            r#"
            MATCH (entry:Function {{name: $name}})<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p:Project {{id: $project_id}})
            OPTIONAL MATCH (entry)-[:CALLS*1..{depth}]->(callee:Function)
            WHERE EXISTS {{ MATCH (callee)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p) }}
            OPTIONAL MATCH (caller:Function)-[:CALLS*1..{depth}]->(entry)
            WHERE EXISTS {{ MATCH (caller)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p) }}
            WITH entry,
                 collect(DISTINCT callee) AS callees_nodes,
                 collect(DISTINCT caller) AS callers_nodes
            WITH entry, callees_nodes, callers_nodes,
                 [entry] + callees_nodes + callers_nodes AS all_funcs
            UNWIND all_funcs AS f
            WITH DISTINCT f
            WHERE f IS NOT NULL
            RETURN f.name AS func_name, f.file_path AS file_path,
                   f.community_id AS community_id
            "#,
            depth = depth
        ))
        .param("name", entry_function)
        .param("project_id", project_id.to_string());

        let rows = self.execute_with_params(q).await?;

        // Collect unique functions with community info
        let mut functions: Vec<(String, Option<String>, Option<i64>)> = Vec::new();
        let mut files: std::collections::HashSet<String> = std::collections::HashSet::new();

        for row in &rows {
            if let Ok(func_name) = row.get::<String>("func_name") {
                let file_path: Option<String> = row.get::<String>("file_path").ok();
                let community_id: Option<i64> = row.get::<i64>("community_id").ok();
                if let Some(ref fp) = file_path {
                    if !fp.is_empty() {
                        files.insert(fp.clone());
                    }
                }
                functions.push((func_name, file_path, community_id));
            }
        }

        if functions.is_empty() {
            return Err(anyhow::anyhow!(
                "No function found matching '{}'",
                entry_function
            ));
        }

        // Step 1a: Community-based filtering (post-traversal)
        // Keep same community OR direct (depth=1) connection OR no community data
        if filter_community {
            let entry_community = functions
                .iter()
                .find(|(name, _, _)| name == entry_function)
                .and_then(|(_, _, cid)| *cid);

            if let Some(entry_cid) = entry_community {
                // Get direct (depth=1) callees and callers for the entry function
                let direct_q = query(
                    r#"
                    MATCH (entry:Function {name: $name})<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p:Project {id: $project_id})
                    OPTIONAL MATCH (entry)-[:CALLS]->(dc:Function)
                    WHERE EXISTS { MATCH (dc)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p) }
                    OPTIONAL MATCH (caller:Function)-[:CALLS]->(entry)
                    WHERE EXISTS { MATCH (caller)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p) }
                    WITH collect(DISTINCT dc.name) AS direct_callees,
                         collect(DISTINCT caller.name) AS direct_callers
                    RETURN direct_callees + direct_callers AS direct_names
                    "#,
                )
                .param("name", entry_function)
                .param("project_id", project_id.to_string());

                let direct_rows = self.execute_with_params(direct_q).await?;
                let mut direct_set: std::collections::HashSet<String> =
                    std::collections::HashSet::new();
                if let Some(row) = direct_rows.first() {
                    if let Ok(names) = row.get::<Vec<String>>("direct_names") {
                        for n in names {
                            if !n.is_empty() {
                                direct_set.insert(n);
                            }
                        }
                    }
                }

                // Filter: keep entry + same community + direct connections + unknown community
                functions.retain(|(fname, _, cid)| {
                    fname == entry_function
                        || direct_set.contains(fname)
                        || cid.is_none()
                        || *cid == Some(entry_cid)
                });

                // Rebuild files set from remaining functions
                files.clear();
                for (_, fp, _) in &functions {
                    if let Some(ref fp) = fp {
                        if !fp.is_empty() {
                            files.insert(fp.clone());
                        }
                    }
                }
            }
            // If entry has no community_id → no filtering (backward compat)
        }

        // Step 1b: Expand via IMPLEMENTS_TRAIT + IMPLEMENTS_FOR for structs/enums in collected files
        let mut structs: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut traits: std::collections::HashSet<String> = std::collections::HashSet::new();

        if !files.is_empty()
            && (should_include("implements_trait") || should_include("implements_for"))
        {
            let file_list: Vec<String> = files.iter().cloned().collect();
            let types_q = query(
                r#"
                MATCH (f:File)<-[:CONTAINS]-(p:Project {id: $project_id})
                WHERE f.path IN $files
                OPTIONAL MATCH (f)-[:CONTAINS]->(s:Struct)
                OPTIONAL MATCH (f)-[:CONTAINS]->(e:Enum)
                WITH collect(DISTINCT s) + collect(DISTINCT e) AS types
                UNWIND types AS t
                WITH DISTINCT t WHERE t IS NOT NULL
                OPTIONAL MATCH (impl:Impl)-[:IMPLEMENTS_FOR]->(t)
                OPTIONAL MATCH (impl)-[:IMPLEMENTS_TRAIT]->(tr:Trait)
                RETURN t.name AS type_name, labels(t)[0] AS type_label,
                       collect(DISTINCT tr.name) AS trait_names
                "#,
            )
            .param("project_id", project_id.to_string())
            .param("files", file_list);

            let type_rows = self.execute_with_params(types_q).await?;
            for row in &type_rows {
                if let Ok(type_name) = row.get::<String>("type_name") {
                    structs.insert(type_name);
                }
                if let Ok(trait_list) = row.get::<Vec<String>>("trait_names") {
                    for t in trait_list {
                        if !t.is_empty() && !Self::is_blacklisted_trait(&t) {
                            traits.insert(t);
                        }
                    }
                }
            }
        }

        // Step 1c: Expand via IMPORTS — include files imported by the feature's files
        if !files.is_empty() && should_include("imports") {
            let file_list: Vec<String> = files.iter().cloned().collect();
            let imports_q = query(
                r#"
                MATCH (f:File)-[:IMPORTS]->(imported:File)<-[:CONTAINS]-(p:Project {id: $project_id})
                WHERE f.path IN $files
                RETURN DISTINCT imported.path AS imported_path
                "#,
            )
            .param("project_id", project_id.to_string())
            .param("files", file_list);

            let import_rows = self.execute_with_params(imports_q).await?;
            for row in &import_rows {
                if let Ok(imported_path) = row.get::<String>("imported_path") {
                    if !imported_path.is_empty() {
                        files.insert(imported_path);
                    }
                }
            }
        }

        // Step 1d: Expand via vector similarity — find semantically similar functions (best-effort)
        // Uses HNSW vector search on function embeddings, filtered by community if enabled
        // Functions added here get role "support" instead of "core_logic"
        let mut vector_expanded_funcs: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        {
            let func_names: Vec<String> = functions.iter().map(|(n, _, _)| n.clone()).collect();
            let entry_cid = functions
                .iter()
                .find(|(n, _, _)| n == entry_function)
                .and_then(|(_, _, cid)| *cid);
            let community_filter = if filter_community { entry_cid } else { None };

            let vector_neighbors = self
                .expand_by_vector_similarity(&func_names, project_id, community_filter)
                .await;

            for (func_name, file_path, _score) in vector_neighbors {
                if !functions.iter().any(|(n, _, _)| n == &func_name) {
                    if !file_path.is_empty() {
                        files.insert(file_path.clone());
                    }
                    vector_expanded_funcs.insert(func_name.clone());
                    functions.push((func_name, Some(file_path), None));
                }
            }
        }

        // Step 1e: Expand via community — find related files in the same Louvain cluster (best-effort)
        // For each file's community, adds top-5 peers by PageRank that aren't already included
        {
            let file_list: Vec<String> = files.iter().cloned().collect();
            let community_files = self.expand_by_community(&file_list, project_id, 5).await;

            for file_path in community_files {
                files.insert(file_path);
            }
        }

        // Step 2: Create the FeatureGraph (with build params for refresh)
        let fg = FeatureGraphNode {
            id: Uuid::new_v4(),
            name: name.to_string(),
            description: description.map(|d| d.to_string()),
            project_id,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            entity_count: None,
            entry_function: Some(entry_function.to_string()),
            build_depth: Some(depth),
            include_relations: include_relations.map(|r| r.iter().map(|s| s.to_string()).collect()),
        };
        self.create_feature_graph(&fg).await?;

        // Step 3: Add all entities with auto-assigned roles
        let mut entities = Vec::new();

        // Add functions with role: entry_point for the entry function,
        // support for vector-expanded functions, core_logic for others
        for (func_name, _file_path, _) in &functions {
            let role = if func_name == entry_function {
                "entry_point"
            } else if vector_expanded_funcs.contains(func_name) {
                "support"
            } else {
                "core_logic"
            };
            let _ = self
                .add_entity_to_feature_graph(
                    fg.id,
                    "function",
                    func_name,
                    Some(role),
                    Some(project_id),
                )
                .await;
            entities.push(FeatureGraphEntity {
                entity_type: "function".to_string(),
                entity_id: func_name.clone(),
                name: Some(func_name.clone()),
                role: Some(role.to_string()),
                importance_score: None,
            });
        }

        // Add files with role: support
        for file_path in &files {
            let _ = self
                .add_entity_to_feature_graph(
                    fg.id,
                    "file",
                    file_path,
                    Some("support"),
                    Some(project_id),
                )
                .await;
            entities.push(FeatureGraphEntity {
                entity_type: "file".to_string(),
                entity_id: file_path.clone(),
                name: Some(file_path.clone()),
                role: Some("support".to_string()),
                importance_score: None,
            });
        }

        // Add structs/enums discovered via IMPLEMENTS_FOR with role: data_model
        for struct_name in &structs {
            let _ = self
                .add_entity_to_feature_graph(
                    fg.id,
                    "struct",
                    struct_name,
                    Some("data_model"),
                    Some(project_id),
                )
                .await;
            entities.push(FeatureGraphEntity {
                entity_type: "struct".to_string(),
                entity_id: struct_name.clone(),
                name: Some(struct_name.clone()),
                role: Some("data_model".to_string()),
                importance_score: None,
            });
        }

        // Add traits discovered via IMPLEMENTS_TRAIT with role: trait_contract
        for trait_name in &traits {
            let _ = self
                .add_entity_to_feature_graph(
                    fg.id,
                    "trait",
                    trait_name,
                    Some("trait_contract"),
                    Some(project_id),
                )
                .await;
            entities.push(FeatureGraphEntity {
                entity_type: "trait".to_string(),
                entity_id: trait_name.clone(),
                name: Some(trait_name.clone()),
                role: Some("trait_contract".to_string()),
                importance_score: None,
            });
        }

        // Relations will be populated when fetching via get_feature_graph_detail
        Ok(FeatureGraphDetail {
            graph: fg,
            entities,
            relations: vec![],
        })
    }

    /// Refresh an auto-built feature graph by re-running BFS with stored params.
    /// Returns None if the graph was manually created (no entry_function).
    pub async fn refresh_feature_graph(&self, id: Uuid) -> Result<Option<FeatureGraphDetail>> {
        // 1. Load the existing feature graph
        let detail = self
            .get_feature_graph_detail(id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Feature graph {} not found", id))?;

        let fg = &detail.graph;

        // 2. Check if it was auto-built (has entry_function)
        let entry_function = match &fg.entry_function {
            Some(ef) => ef.clone(),
            None => return Ok(None), // manually created, skip refresh
        };

        let depth = fg.build_depth.unwrap_or(2);
        let include_relations = fg.include_relations.clone();
        let project_id = fg.project_id;

        // 3. Delete all existing INCLUDES_ENTITY relationships
        let delete_q = query(
            "MATCH (fg:FeatureGraph {id: $fg_id})-[r:INCLUDES_ENTITY]->()
             DELETE r
             RETURN count(r) AS deleted",
        )
        .param("fg_id", id.to_string());
        self.execute_with_params(delete_q).await?;

        // 4. Re-run the BFS traversal (same logic as auto_build_feature_graph)
        let should_include = |rel: &str| -> bool {
            match &include_relations {
                None => true,
                Some(rels) => rels.iter().any(|r| r.eq_ignore_ascii_case(rel)),
            }
        };

        let depth = depth.clamp(1, 5);

        // Step 4a: Collect functions via call graph
        let q = query(&format!(
            r#"
            MATCH (entry:Function {{name: $name}})<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p:Project {{id: $project_id}})
            OPTIONAL MATCH (entry)-[:CALLS*1..{depth}]->(callee:Function)
            WHERE EXISTS {{ MATCH (callee)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p) }}
            OPTIONAL MATCH (caller:Function)-[:CALLS*1..{depth}]->(entry)
            WHERE EXISTS {{ MATCH (caller)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p) }}
            WITH entry,
                 collect(DISTINCT callee) AS callees_nodes,
                 collect(DISTINCT caller) AS callers_nodes
            WITH entry, callees_nodes, callers_nodes,
                 [entry] + callees_nodes + callers_nodes AS all_funcs
            UNWIND all_funcs AS f
            WITH DISTINCT f
            WHERE f IS NOT NULL
            RETURN f.name AS func_name, f.file_path AS file_path
            "#,
            depth = depth
        ))
        .param("name", entry_function.as_str())
        .param("project_id", project_id.to_string());

        let rows = self.execute_with_params(q).await?;

        let mut functions: Vec<(String, Option<String>)> = Vec::new();
        let mut files: std::collections::HashSet<String> = std::collections::HashSet::new();

        for row in &rows {
            if let Ok(func_name) = row.get::<String>("func_name") {
                let file_path: Option<String> = row.get::<String>("file_path").ok();
                if let Some(ref fp) = file_path {
                    if !fp.is_empty() {
                        files.insert(fp.clone());
                    }
                }
                functions.push((func_name, file_path));
            }
        }

        if functions.is_empty() {
            return Err(anyhow::anyhow!(
                "No function found matching '{}' during refresh",
                entry_function
            ));
        }

        // Step 4b: Expand via IMPLEMENTS_TRAIT + IMPLEMENTS_FOR
        let mut structs: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut traits: std::collections::HashSet<String> = std::collections::HashSet::new();

        if !files.is_empty()
            && (should_include("implements_trait") || should_include("implements_for"))
        {
            let file_list: Vec<String> = files.iter().cloned().collect();
            let types_q = query(
                r#"
                MATCH (f:File)<-[:CONTAINS]-(p:Project {id: $project_id})
                WHERE f.path IN $files
                OPTIONAL MATCH (f)-[:CONTAINS]->(s:Struct)
                OPTIONAL MATCH (f)-[:CONTAINS]->(e:Enum)
                WITH collect(DISTINCT s) + collect(DISTINCT e) AS types
                UNWIND types AS t
                WITH DISTINCT t WHERE t IS NOT NULL
                OPTIONAL MATCH (impl:Impl)-[:IMPLEMENTS_FOR]->(t)
                OPTIONAL MATCH (impl)-[:IMPLEMENTS_TRAIT]->(tr:Trait)
                RETURN t.name AS type_name, labels(t)[0] AS type_label,
                       collect(DISTINCT tr.name) AS trait_names
                "#,
            )
            .param("project_id", project_id.to_string())
            .param("files", file_list);

            let type_rows = self.execute_with_params(types_q).await?;
            for row in &type_rows {
                if let Ok(type_name) = row.get::<String>("type_name") {
                    structs.insert(type_name);
                }
                if let Ok(trait_list) = row.get::<Vec<String>>("trait_names") {
                    for t in trait_list {
                        if !t.is_empty() && !Self::is_blacklisted_trait(&t) {
                            traits.insert(t);
                        }
                    }
                }
            }
        }

        // Step 4c: Expand via IMPORTS
        if !files.is_empty() && should_include("imports") {
            let file_list: Vec<String> = files.iter().cloned().collect();
            let imports_q = query(
                r#"
                MATCH (f:File)-[:IMPORTS]->(imported:File)<-[:CONTAINS]-(p:Project {id: $project_id})
                WHERE f.path IN $files
                RETURN DISTINCT imported.path AS imported_path
                "#,
            )
            .param("project_id", project_id.to_string())
            .param("files", file_list);

            let import_rows = self.execute_with_params(imports_q).await?;
            for row in &import_rows {
                if let Ok(imported_path) = row.get::<String>("imported_path") {
                    if !imported_path.is_empty() {
                        files.insert(imported_path);
                    }
                }
            }
        }

        // 5. Update the updated_at timestamp
        let update_q = query(
            "MATCH (fg:FeatureGraph {id: $fg_id})
             SET fg.updated_at = $now
             RETURN fg",
        )
        .param("fg_id", id.to_string())
        .param("now", chrono::Utc::now().to_rfc3339());
        self.execute_with_params(update_q).await?;

        // 6. Re-add all entities with auto-assigned roles
        let mut entities = Vec::new();

        for (func_name, _file_path) in &functions {
            let role = if *func_name == entry_function {
                "entry_point"
            } else {
                "core_logic"
            };
            let _ = self
                .add_entity_to_feature_graph(
                    id,
                    "function",
                    func_name,
                    Some(role),
                    Some(project_id),
                )
                .await;
            entities.push(FeatureGraphEntity {
                entity_type: "function".to_string(),
                entity_id: func_name.clone(),
                name: Some(func_name.clone()),
                role: Some(role.to_string()),
                importance_score: None,
            });
        }

        for file_path in &files {
            let _ = self
                .add_entity_to_feature_graph(
                    id,
                    "file",
                    file_path,
                    Some("support"),
                    Some(project_id),
                )
                .await;
            entities.push(FeatureGraphEntity {
                entity_type: "file".to_string(),
                entity_id: file_path.clone(),
                name: Some(file_path.clone()),
                role: Some("support".to_string()),
                importance_score: None,
            });
        }

        for struct_name in &structs {
            let _ = self
                .add_entity_to_feature_graph(
                    id,
                    "struct",
                    struct_name,
                    Some("data_model"),
                    Some(project_id),
                )
                .await;
            entities.push(FeatureGraphEntity {
                entity_type: "struct".to_string(),
                entity_id: struct_name.clone(),
                name: Some(struct_name.clone()),
                role: Some("data_model".to_string()),
                importance_score: None,
            });
        }

        for trait_name in &traits {
            let _ = self
                .add_entity_to_feature_graph(
                    id,
                    "trait",
                    trait_name,
                    Some("trait_contract"),
                    Some(project_id),
                )
                .await;
            entities.push(FeatureGraphEntity {
                entity_type: "trait".to_string(),
                entity_id: trait_name.clone(),
                name: Some(trait_name.clone()),
                role: Some("trait_contract".to_string()),
                importance_score: None,
            });
        }

        // Re-read the graph node to get updated state
        let updated_graph = self
            .get_feature_graph_detail(id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Feature graph {} disappeared after refresh", id))?;

        Ok(Some(updated_graph))
    }

    /// Get the top N most connected functions for a project, ranked by
    /// (callers + callees). Used for auto-generating feature graphs after sync.
    pub async fn get_top_entry_functions(
        &self,
        project_id: Uuid,
        limit: usize,
    ) -> Result<Vec<String>> {
        let q = query(
            r#"
            MATCH (f:Function)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p:Project {id: $project_id})
            OPTIONAL MATCH (f)-[:CALLS]->(callee:Function)
            OPTIONAL MATCH (caller:Function)-[:CALLS]->(f)
            WITH f, count(DISTINCT callee) AS callees, count(DISTINCT caller) AS callers
            WITH f, callers + callees AS connections
            WHERE connections > 0
            RETURN f.name AS name
            ORDER BY connections DESC
            LIMIT $limit
            "#,
        )
        .param("project_id", project_id.to_string())
        .param("limit", limit as i64);

        let mut result = self.graph.execute(q).await?;
        let mut functions = Vec::new();

        while let Some(row) = result.next().await? {
            if let Ok(name) = row.get::<String>("name") {
                functions.push(name);
            }
        }

        Ok(functions)
    }

    fn node_to_feature_graph(&self, node: &neo4rs::Node) -> Result<FeatureGraphNode> {
        Ok(FeatureGraphNode {
            id: node.get::<String>("id")?.parse()?,
            name: node.get("name")?,
            description: node.get::<String>("description").ok().and_then(|s| {
                if s.is_empty() {
                    None
                } else {
                    Some(s)
                }
            }),
            project_id: node.get::<String>("project_id")?.parse()?,
            created_at: node
                .get::<String>("created_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            updated_at: node
                .get::<String>("updated_at")?
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            entity_count: None,
            entry_function: node.get::<String>("entry_function").ok(),
            build_depth: node.get::<i64>("build_depth").ok().map(|d| d as u32),
            include_relations: node
                .get::<String>("include_relations")
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok()),
        })
    }
}
