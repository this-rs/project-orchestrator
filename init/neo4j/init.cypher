// Neo4j initialization script
// This is run when the database is first created

// ============================================================================
// Constraints (ensure uniqueness)
// ============================================================================

CREATE CONSTRAINT file_path IF NOT EXISTS FOR (f:File) REQUIRE f.path IS UNIQUE;
CREATE CONSTRAINT function_id IF NOT EXISTS FOR (f:Function) REQUIRE f.id IS UNIQUE;
CREATE CONSTRAINT struct_id IF NOT EXISTS FOR (s:Struct) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT trait_id IF NOT EXISTS FOR (t:Trait) REQUIRE t.id IS UNIQUE;
CREATE CONSTRAINT enum_id IF NOT EXISTS FOR (e:Enum) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT plan_id IF NOT EXISTS FOR (p:Plan) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT task_id IF NOT EXISTS FOR (t:Task) REQUIRE t.id IS UNIQUE;
CREATE CONSTRAINT step_id IF NOT EXISTS FOR (s:Step) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT decision_id IF NOT EXISTS FOR (d:Decision) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT constraint_id IF NOT EXISTS FOR (c:Constraint) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT agent_id IF NOT EXISTS FOR (a:Agent) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT commit_hash IF NOT EXISTS FOR (c:Commit) REQUIRE c.hash IS UNIQUE;

// ============================================================================
// Indexes (improve query performance)
// ============================================================================

// Code structure indexes
CREATE INDEX file_language IF NOT EXISTS FOR (f:File) ON (f.language);
CREATE INDEX file_hash IF NOT EXISTS FOR (f:File) ON (f.hash);
CREATE INDEX function_name IF NOT EXISTS FOR (f:Function) ON (f.name);
CREATE INDEX function_file IF NOT EXISTS FOR (f:Function) ON (f.file_path);
CREATE INDEX struct_name IF NOT EXISTS FOR (s:Struct) ON (s.name);
CREATE INDEX trait_name IF NOT EXISTS FOR (t:Trait) ON (t.name);

// Plan indexes
CREATE INDEX plan_status IF NOT EXISTS FOR (p:Plan) ON (p.status);
CREATE INDEX plan_priority IF NOT EXISTS FOR (p:Plan) ON (p.priority);
CREATE INDEX task_status IF NOT EXISTS FOR (t:Task) ON (t.status);
CREATE INDEX task_assigned IF NOT EXISTS FOR (t:Task) ON (t.assigned_to);
CREATE INDEX step_status IF NOT EXISTS FOR (s:Step) ON (s.status);
CREATE INDEX decision_timestamp IF NOT EXISTS FOR (d:Decision) ON (d.decided_at);

// Agent indexes
CREATE INDEX agent_type IF NOT EXISTS FOR (a:Agent) ON (a.agent_type);
CREATE INDEX agent_active IF NOT EXISTS FOR (a:Agent) ON (a.last_active);

// ============================================================================
// Full-text indexes (for text search within Neo4j)
// ============================================================================

// Note: These are created only if they don't exist
// CALL db.index.fulltext.createNodeIndex("functionSearch", ["Function"], ["name", "docstring"]);
// CALL db.index.fulltext.createNodeIndex("decisionSearch", ["Decision"], ["description", "rationale"]);
