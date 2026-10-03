<!-- Genere par scripts/diagrams/check-index.mjs --write-orphans. Ne pas editer a la main. -->
<!-- orphan-ceiling: 1119 -->
<!-- orphan-ceiling-backend: 446 -->
<!-- orphan-ceiling-frontend: 542 -->
<!-- orphan-ceiling-nexus: 4 -->
<!-- orphan-ceiling-website: 127 -->

# Fichiers source sans diagramme proprietaire

Un fichier source est **orphelin** quand aucun `covers` de `INDEX.yml` ne le matche :
aucun diagramme ne repond de son comportement. Cette liste est publiee pour etre honnete
sur ce que la cartographie couvre reellement — on ne reduit pas le denominateur, on la reduit elle.

**1119 orphelins sur 1253 fichiers source (89.3 %).**

SEULE une entree `verified` possede un fichier. Une entree `planned` annonce un perimetre
sans qu'un diagramme existe : compter ses globs ferait baisser ce nombre sans qu'une ligne
soit ecrite, et l'index acheterait du credit sur des intentions.
Sur les 1119 orphelins, **440 sont deja reserves** par une entree `planned` :
leur proprietaire est designe, son diagramme reste a ecrire.

Le plafond est **1119** : le verificateur echoue si le nombre reel le depasse.
Il ne peut que descendre. Ajouter un fichier source sans proprietaire fait echouer la build ;
la sortie est un glob `covers`, pas un plafond plus haut. `--raise-ceiling` existe mais exige
une raison ecrite, et un plafond releve se voit dans la revue.

Regeneration (hors reseau) :

```
node scripts/diagrams/check-index.mjs --write-orphans
```

Un depot voisin peut tenir son propre index pour les diagrammes dont le `.mmd` vit chez lui.
Les fichiers qu'il possede ont un proprietaire et ne figurent donc pas ci-dessous ; toute
collision entre les deux index est une erreur, pas un arrangement.

- `nexus` : 14 diagrammes, 77 fichiers possedes la-bas

## backend (446)

- `crates/neural-routing-core/src/` (12) : `augmentation.rs`, `error.rs`, `lib.rs`, `mcts.rs`, `migration.rs`, `models.rs`, `proxy_model.rs`, `reward.rs`, `store.rs`, `traits.rs`, `validation.rs`, `vector_builder.rs`
- `crates/neural-routing-gnn/src/` (9) : `encoder.rs`, `features.rs`, `graph_sage.rs`, `inference.rs`, `lib.rs`, `message_passing.rs`, `rgcn.rs`, `sampler.rs`, `training.rs`
- `crates/neural-routing-gnn/src/benchmark/` (3) : `metrics.rs`, `mod.rs`, `statistical.rs`
- `crates/neural-routing-nn/src/` (5) : `benchmark.rs`, `lib.rs`, `metrics.rs`, `router.rs`, `scoring.rs`
- `crates/neural-routing-policy/src/` (13) : `ab_testing.rs`, `action_decoder.rs`, `benchmark.rs`, `codebook.rs`, `cql.rs`, `dataloader.rs`, `dataset.rs`, `evaluation.rs`, `ewc.rs`, `lib.rs`, `registry.rs`, `training.rs`, `transformer.rs`
- `crates/neural-routing-runtime/src/` (11) : `collector.rs`, `confidence.rs`, `config.rs`, `continual.rs`, `cpu_guard.rs`, `drift.rs`, `dual_track.rs`, `exploration.rs`, `inference_engine.rs`, `lib.rs`, `reward.rs`
- `crates/tree-sitter-dart/src/` (1) : `lib.rs`
- `crates/tree-sitter-hcl/src/` (1) : `lib.rs`
- `desktop/src-tauri/src/` (5) : `docker.rs`, `main.rs`, `setup.rs`, `tray.rs`, `updater.rs`
- `desktop/src-tauri/src/plugins/` (2) : `mac_rounded_corners.rs`, `mod.rs`
- `src/` (5) : `cli.rs`, `homeostasis.rs`, `lib.rs`, `main.rs`, `setup_claude.rs`
- `src/analytics/` (3) : `distribution.rs`, `hypothesis.rs`, `mod.rs`
- `src/analytics/stats/` (5) : `anova.rs`, `fitting.rs`, `golden_fixtures.rs`, `mean_std.rs`, `mod.rs`
- `src/api/` (36) : `attention.rs`, `attention_aggregate.rs`, `auth_handlers.rs`, `chat_handlers.rs`, `code_handlers.rs`, `document_handlers.rs`, `embedded_frontend.rs`, `environment_handlers.rs`, `episode_handlers.rs`, `feedback_handlers.rs`, `graph_handlers.rs`, `graph_types.rs`, `handlers.rs`, `hook_handlers.rs`, `mcp_federation_handlers.rs`, `mod.rs`, `neural_routing_handlers.rs`, `note_handlers.rs`, `persona_handlers.rs`, `profile_handlers.rs`, `project_handlers.rs`, `protocol_handlers.rs`, `query.rs`, `reason_handlers.rs`, `registry_handlers.rs`, `rfc_handlers.rs`, `routes.rs`, `sharing_handlers.rs`, `skill_handlers.rs`, `trajectory_handlers.rs`, `trigger_handlers.rs`, `vault_handlers.rs`, `workspace_handlers.rs`, `ws_auth.rs`, `ws_handlers.rs`, `ws_run_handler.rs`
- `src/architecture/` (7) : `catalogue.rs`, `compose.rs`, `derive.rs`, `manifest.rs`, `mod.rs`, `runtime_config.rs`, `sync.rs`
- `src/auth/` (8) : `extractor.rs`, `google.rs`, `jwt.rs`, `middleware.rs`, `mod.rs`, `oauth_server.rs`, `oidc.rs`, `refresh.rs`
- `src/bin/` (1) : `mcp_server.rs`
- `src/chat/` (14) : `attachment.rs`, `attention.rs`, `composer.rs`, `continuity.rs`, `enrichment.rs`, `entity_extractor.rs`, `feedback.rs`, `observation_detector.rs`, `path_detect.rs`, `prompt.rs`, `prompt_sections.rs`, `routing.rs`, `viz.rs`, `viz_builder.rs`
- `src/chat/stages/` (10) : `biomimicry.rs`, `file_context.rs`, `intent_weights.rs`, `knowledge_injection.rs`, `mcp_federation_stage.rs`, `mod.rs`, `persona.rs`, `skill_activation.rs`, `status_injection.rs`, `user_profile.rs`
- `src/documents/` (4) : `align.rs`, `chunk.rs`, `mod.rs`, `store.rs`
- `src/documents/extract/` (7) : `docx.rs`, `mod.rs`, `pdf.rs`, `pptx.rs`, `text.rs`, `xlsx.rs`, `xml.rs`
- `src/embeddings/` (5) : `fastembed.rs`, `mock.rs`, `mod.rs`, `provider.rs`, `traits.rs`
- `src/episodes/` (8) : `anonymize.rs`, `artifact_comparison.rs`, `collector.rs`, `distill.rs`, `distill_models.rs`, `evaluation.rs`, `mod.rs`, `models.rs`
- `src/events/` (13) : `attention.rs`, `builtin_triggers.rs`, `bus.rs`, `graph.rs`, `hybrid.rs`, `mod.rs`, `nats.rs`, `notifier.rs`, `reactions.rs`, `reactor.rs`, `trigger.rs`, `trigger_routing.rs`, `types.rs`
- `src/feedback/` (6) : `handlers.rs`, `mod.rs`, `models.rs`, `propagator.rs`, `signals.rs`, `tracker.rs`
- `src/graph/` (12) : `algorithms.rs`, `confidence.rs`, `debouncer.rs`, `engine.rs`, `enrichment.rs`, `extraction.rs`, `mock.rs`, `mod.rs`, `models.rs`, `neighborhood.rs`, `process.rs`, `writer.rs`
- `src/heartbeat/` (2) : `engine.rs`, `mod.rs`
- `src/heartbeat/checks/` (10) : `architecture_drift.rs`, `consolidation.rs`, `convention_guard.rs`, `git_drift.rs`, `homeostasis.rs`, `maintenance.rs`, `mod.rs`, `staleness.rs`, `synapse_decay.rs`, `synapse_replenish.rs`
- `src/identity/` (4) : `did.rs`, `mod.rs`, `rotation.rs`, `token.rs`
- `src/lifecycle/` (3) : `executor.rs`, `mod.rs`, `models.rs`
- `src/mcp/` (9) : `formatter.rs`, `handlers.rs`, `http_client.rs`, `http_transport.rs`, `mod.rs`, `pipeline_handler.rs`, `protocol.rs`, `server.rs`, `tools.rs`
- `src/mcp_federation/` (7) : `circuit_breaker.rs`, `client.rs`, `discovery.rs`, `mod.rs`, `prober.rs`, `registry.rs`, `security.rs`
- `src/meilisearch/` (6) : `client.rs`, `impl_search_store.rs`, `indexes.rs`, `mock.rs`, `mod.rs`, `traits.rs`
- `src/neo4j/` (42) : `agent_execution.rs`, `alert.rs`, `analytics.rs`, `batch.rs`, `chat.rs`, `client.rs`, `code.rs`, `commit.rs`, `constraint.rs`, `data_migrations.rs`, `decision.rs`, `document.rs`, `environment.rs`, `event_trigger.rs`, `feature_graph.rs`, `impl_graph_store.rs`, `lifecycle_hook.rs`, `mcp_federation.rs`, `milestone.rs`, `mock.rs`, `mod.rs`, `models.rs`, `neighborhood.rs`, `note.rs`, `persona.rs`, `plan.rs`, `plan_run.rs`, `profile.rs`, `project.rs`, `protocol.rs`, `reasoning.rs`, `registry.rs`, `release.rs`, `sharing.rs`, `skill.rs`, `step.rs`, `task.rs`, `topology.rs`, `traits.rs`, `trigger.rs`, `user.rs`, `workspace.rs`
- `src/neurons/` (5) : `activation.rs`, `config.rs`, `intent.rs`, `mod.rs`, `search.rs`
- `src/notes/` (6) : `hashing.rs`, `lifecycle.rs`, `manager.rs`, `mod.rs`, `models.rs`, `witness.rs`
- `src/orchestrator/` (6) : `context.rs`, `mod.rs`, `planner.rs`, `runner.rs`, `topology_hook.rs`, `watcher.rs`
- `src/parser/` (4) : `ast_cache.rs`, `helpers.rs`, `mod.rs`, `noise_filter.rs`
- `src/parser/languages/` (18) : `bash.rs`, `c.rs`, `cpp.rs`, `csharp.rs`, `dart.rs`, `go.rs`, `hcl.rs`, `java.rs`, `kotlin.rs`, `mod.rs`, `php.rs`, `python.rs`, `ruby.rs`, `rust.rs`, `scala.rs`, `swift.rs`, `typescript.rs`, `zig.rs`
- `src/pipeline/` (14) : `composer.rs`, `critic.rs`, `episode_adapter.rs`, `evolve.rs`, `feedback.rs`, `gates.rs`, `materialize.rs`, `metrics.rs`, `mod.rs`, `progress.rs`, `regression.rs`, `runner.rs`, `skill_injector.rs`, `wave_executor.rs`
- `src/plan/` (3) : `manager.rs`, `mod.rs`, `models.rs`
- `src/profile/` (5) : `aggregator.rs`, `collector.rs`, `mod.rs`, `signals.rs`, `wiring.rs`
- `src/protocol/` (9) : `engine.rs`, `generator.rs`, `hooks.rs`, `mod.rs`, `models.rs`, `routing.rs`, `runner.rs`, `seed.rs`, `seed_runner.rs`
- `src/protocol/executor/` (3) : `agent.rs`, `mod.rs`, `system.rs`
- `src/reasoning/` (4) : `cache.rs`, `engine.rs`, `mod.rs`, `models.rs`
- `src/reception/` (7) : `anchor.rs`, `mod.rs`, `replay.rs`, `score.rs`, `tombstone_scheduler.rs`, `trust.rs`, `verify.rs`
- `src/reflex/` (5) : `co_change.rs`, `episode_recall.rs`, `mod.rs`, `scar_warning.rs`, `stage.rs`
- `src/resolver/` (4) : `mod.rs`, `resolve_cache.rs`, `suffix_index.rs`, `symbol_table.rs`
- `src/runner/` (16) : `eligibility.rs`, `enricher.rs`, `feedback.rs`, `feedback_analyzer.rs`, `git.rs`, `guard.rs`, `lifecycle.rs`, `mod.rs`, `models.rs`, `persona.rs`, `prompt.rs`, `runner.rs`, `state.rs`, `trigger.rs`, `vector.rs`, `verifier.rs`
- `src/runner/providers/` (4) : `event.rs`, `mod.rs`, `schedule.rs`, `webhook.rs`
- `src/sharing/` (5) : `consent_gate.rs`, `mod.rs`, `revocation.rs`, `tombstone.rs`, `ttl.rs`
- `src/skills/` (20) : `activation.rs`, `cache.rs`, `detection.rs`, `evolution.rs`, `export.rs`, `feedback.rs`, `hook_extractor.rs`, `import.rs`, `lifecycle.rs`, `maintenance.rs`, `mod.rs`, `models.rs`, `naming.rs`, `package.rs`, `project_resolver.rs`, `registry.rs`, `templates.rs`, `triggers.rs`, `trust.rs`, `validation.rs`
- `src/transport/` (5) : `http.rs`, `http_handlers.rs`, `mod.rs`, `sync.rs`, `types.rs`
- `src/update/` (4) : `deployment.rs`, `mod.rs`, `service.rs`, `version.rs`
- `src/utils/` (3) : `file_path_extractor.rs`, `mod.rs`, `paths.rs`
- `src/vault/` (7) : `agent_cli.rs`, `crypto.rs`, `grants.rs`, `mask.rs`, `mod.rs`, `service.rs`, `store.rs`

## frontend (542)

- `src/` (2) : `App.tsx`, `main.tsx`
- `src/adapters/` (3) : `MilestoneGraphAdapter.ts`, `PlanGraphAdapter.ts`, `TaskGraphAdapter.ts`
- `src/atoms/` (13) : `attentionCount.ts`, `attentionDigest.ts`, `auth.ts`, `events.ts`, `index.ts`, `intelligence.ts`, `notes.ts`, `plans.ts`, `projects.ts`, `setup.ts`, `tasks.ts`, `ui.ts`, `workspaces.ts`
- `src/components/` (8) : `AttentionBadge.tsx`, `DependencyGraphView.tsx`, `GlobalRouteLayout.tsx`, `SetupGuard.tsx`, `TitleBar.tsx`, `UpdateBanner.tsx`, `WorkspaceRouteGuard.tsx`, `WorkspaceSwitcher.tsx`
- `src/components/auth/` (4) : `PasswordLoginForm.tsx`, `ProtectedRoute.tsx`, `RegisterForm.tsx`, `UserMenu.tsx`
- `src/components/chat/` (32) : `AgentGroup.tsx`, `AgenticModeBanner.tsx`, `AgenticModePill.tsx`, `AskUserQuestionBlock.tsx`, `Attachments.tsx`, `BackgroundActivityBlock.tsx`, `BackgroundActivityCard.tsx`, `BackgroundTasksIndicator.tsx`, `ChatWelcome.tsx`, `CompactBoundaryBlock.tsx`, `CompactionBanner.tsx`, `ContinueIndicatorBlock.tsx`, `CopyMarkdownButton.tsx`, `DetachedRunsPanel.tsx`, `MarkdownText.tsx`, `MessageQueueBar.tsx`, `ModelChangedBlock.tsx`, `ModelFamilyPicker.tsx`, `ProjectSelect.tsx`, `ResultErrorBlock.tsx`, `ResultMaxTurnsBlock.tsx`, `RetryIndicatorBlock.tsx`, `SecretRequestTray.tsx`, `SessionBreadcrumb.tsx`, `SystemHintBlock.tsx`, `SystemInitBlock.tsx`, `ThinkingBlock.tsx`, `ToolCallBlock.tsx`, `ToolCallGroup.tsx`, `attachmentState.ts`, `index.ts`, `useElapsedMs.ts`
- `src/components/chat/viz/` (10) : `ContextRadarViz.tsx`, `ImpactGraphViz.tsx`, `KnowledgeCardViz.tsx`, `ProgressBarViz.tsx`, `ProtocolRunViz.tsx`, `ReasoningTreeViz.tsx`, `VizBlockRenderer.tsx`, `VizExpandDialog.tsx`, `index.ts`, `registry.ts`
- `src/components/code/` (11) : `CoChangeGraph.tsx`, `CodeArchitectureFullTab.tsx`, `CodeArchitectureTab.tsx`, `CodeCommunitiesTab.tsx`, `CodeExplorerTab.tsx`, `CodeHealthTab.tsx`, `CodeHeritageTab.tsx`, `CodeProcessesTab.tsx`, `CodeSanteTab.tsx`, `FileHistoryDrawer.tsx`, `index.ts`
- `src/components/commits/` (2) : `CommitList.tsx`, `index.ts`
- `src/components/composer/` (6) : `FSMCanvas.tsx`, `NotePool.tsx`, `PatternComposer.tsx`, `TriggerBuilder.tsx`, `index.ts`, `types.ts`
- `src/components/discussions/` (9) : `AttachSessionButton.tsx`, `AttachSessionDialog.tsx`, `DiscussionNode.tsx`, `DiscussionTreeView.tsx`, `InlineConversationPanel.tsx`, `LinkedDiscussions.tsx`, `linkedForest.ts`, `resumeActions.ts`, `useLinkedForest.ts`
- `src/components/expandable/` (1) : `index.tsx`
- `src/components/featureGraphs/` (3) : `EntityBrowser.tsx`, `EntityDetailPanel.tsx`, `FeatureGraphHelp.tsx`
- `src/components/forms/` (26) : `AutoBuildFeatureGraphForm.tsx`, `CreateComponentForm.tsx`, `CreateConstraintForm.tsx`, `CreateDecisionForm.tsx`, `CreateFeatureGraphForm.tsx`, `CreateMilestoneForm.tsx`, `CreateNoteForm.tsx`, `CreatePlanForm.tsx`, `CreateProjectForm.tsx`, `CreateReleaseForm.tsx`, `CreateResourceForm.tsx`, `CreateSkillForm.tsx`, `CreateStepForm.tsx`, `CreateTaskForm.tsx`, `CreateWorkspaceForm.tsx`, `DecisionForms.tsx`, `EditMilestoneForm.tsx`, `EditPersonaForm.tsx`, `EditPlanForm.tsx`, `EditProjectForm.tsx`, `EditStepForm.tsx`, `EditTaskForm.tsx`, `EditWorkspaceForm.tsx`, `ImportSkillForm.tsx`, `NoteForms.tsx`, `index.ts`
- `src/components/graph/` (2) : `EntityGroupPanel.tsx`, `UnifiedGraphSection.tsx`
- `src/components/graph/entity/` (13) : `EntityGraph.tsx`, `EntityGraphCanvas.tsx`, `EntityGraphControls.tsx`, `EntityGraphExplainer.tsx`, `EntityTypeIcon.tsx`, `NodeInfoCard.tsx`, `entityHref.ts`, `entityVisuals.ts`, `index.ts`, `radialLayout.ts`, `useNeighborhood.ts`, `usePanZoom.ts`, `useReducedMotion.ts`
- `src/components/intelligence/` (22) : `ActivityHeatmap3D.tsx`, `ContextRadar.tsx`, `GraphLoadingProgress.tsx`, `IntelligenceDashboard.tsx`, `IntelligenceGraphPage.tsx`, `LayerControls.tsx`, `LearningTimeline.tsx`, `LiveIndicator.tsx`, `NodeInspector.tsx`, `ProtocolRanking.tsx`, `ProtocolRunViewer.tsx`, `SpreadingActivation.tsx`, `VectorSpaceExplorer.tsx`, `WorkspaceGraphPage.tsx`, `WorkspaceLearningTimeline.tsx`, `index.ts`, `useGraphWebSocket.ts`, `useIntelligenceGraph.ts`, `useProtocolRunEvents.ts`, `useWorkspaceIntelligenceData.ts`, `useWorkspaceIntelligenceGraph.ts`, `useWsAnimation.ts`
- `src/components/intelligence/cards/` (4) : `FileContextCard.tsx`, `NoteContextCard.tsx`, `ProtocolContextCard.tsx`, `SkillContextCard.tsx`
- `src/components/intelligence/edges/` (4) : `AffectsEdge.tsx`, `CoChangedEdge.tsx`, `SynapseEdge.tsx`, `index.ts`
- `src/components/intelligence/graph3d/` (6) : `CommunityHulls3D.tsx`, `IntelligenceGraph3D.tsx`, `nodeObjects.ts`, `useActivationSync.ts`, `useGraph3DLayout.ts`, `useRenderLoop.ts`
- `src/components/intelligence/nodes/` (14) : `DecisionNode.tsx`, `EnumNode.tsx`, `FeatureGraphNode.tsx`, `FileNode.tsx`, `FunctionNode.tsx`, `NoteNode.tsx`, `PlanNode.tsx`, `ProtocolNode.tsx`, `ProtocolStateNode.tsx`, `SkillNode.tsx`, `StructNode.tsx`, `TaskNode.tsx`, `TraitNode.tsx`, `index.ts`
- `src/components/intelligence/vectorspace3d/` (1) : `VectorSpace3D.tsx`
- `src/components/kanban/` (11) : `BoardCard.tsx`, `KanbanCard.tsx`, `KanbanFilterBar.tsx`, `ListControls.tsx`, `MilestoneKanbanCard.tsx`, `PlanKanbanCard.tsx`, `PlanKanbanFilterBar.tsx`, `UniversalKanban.tsx`, `UniversalKanbanCard.tsx`, `UniversalKanbanColumn.tsx`, `index.ts`
- `src/components/kanban/configs/` (5) : `milestoneKanbanConfig.tsx`, `planKanbanConfig.tsx`, `stepKanbanConfig.tsx`, `taskKanbanConfig.tsx`, `types.ts`
- `src/components/knowledge/` (4) : `NeuronExplorer.tsx`, `NoteTypeLabel.tsx`, `index.ts`, `noteMeta.ts`
- `src/components/particles/` (2) : `ParticleViz.tsx`, `useParticleEngine.ts`
- `src/components/particles/adapters/` (2) : `index.ts`, `types.ts`
- `src/components/particles/engine/` (6) : `ParticleEngine.ts`, `ParticlePool.ts`, `emitters.ts`, `forces.ts`, `index.ts`, `types.ts`
- `src/components/particles/renderer/` (2) : `CanvasRenderer.ts`, `TextRenderer.ts`
- `src/components/particles/scenes/` (17) : `AttentionScene.ts`, `ContextWindowScene.ts`, `DelegationScene.ts`, `DistributionScene.ts`, `EmbeddingsScene.ts`, `FeedbackLoopScene.ts`, `FineTuningScene.ts`, `FocusScene.ts`, `HumanAIScene.ts`, `LeverageScene.ts`, `MoatScene.ts`, `PromptOutputScene.ts`, `SignalNoiseScene.ts`, `SlideDeckScenes.ts`, `SystemScene.ts`, `index.ts`, `types.ts`
- `src/components/particles/widgets/` (7) : `CommunityVizWidget.tsx`, `ImpactPreviewWidget.tsx`, `ProjectHealthWidget.tsx`, `PropagationVizWidget.tsx`, `ProtocolRunWidget.tsx`, `WaveDispatchWidget.tsx`, `index.ts`
- `src/components/personas/` (2) : `PersonaBuilder.tsx`, `index.ts`
- `src/components/pipeline/` (4) : `ImplementDialog.tsx`, `PipelineNodeRow.tsx`, `PipelineProgressHeader.tsx`, `PipelineTreeView.tsx`
- `src/components/plans/` (3) : `PlanUniverse3D.tsx`, `WaveView.tsx`, `usePlanUniverse.ts`
- `src/components/protocols/` (13) : `Explainer.tsx`, `FsmBreadcrumbs.tsx`, `FsmViewer.tsx`, `GanttTimeline.tsx`, `RecentRunsPanel.tsx`, `RfcDashboardPage.tsx`, `RfcStatusBadge.tsx`, `RunStatusBadge.tsx`, `RunTreeView.tsx`, `ScheduledActionsPanel.tsx`, `index.ts`, `rfcLifecycle.ts`, `runHelpers.ts`
- `src/components/registry/` (7) : `ImportWizard.tsx`, `SkillBrowser.tsx`, `TrustBadge.tsx`, `concepts.tsx`, `fetchAll.ts`, `index.ts`, `metrics.ts`
- `src/components/runner/` (14) : `AgentExecutionDetail.tsx`, `BudgetEditor.tsx`, `CancelButton.tsx`, `InlineConversation.tsx`, `LiveProgress.tsx`, `PlanRunHistory.tsx`, `PlanRunRow.tsx`, `RunnerHeader.tsx`, `StatsRow.tsx`, `WaveAgentCard.tsx`, `WaveSection.tsx`, `WsStatusIndicator.tsx`, `index.ts`, `shared.ts`
- `src/components/settings/` (1) : `SettingRow.tsx`
- `src/components/tasks/` (5) : `DetailRows.tsx`, `RowStateLink.tsx`, `StatusBreakdown.tsx`, `TaskUniverse3D.tsx`, `useTaskUniverse.ts`
- `src/components/today/` (10) : `AttentionCard.tsx`, `ContinueSheet.tsx`, `LaneChips.tsx`, `MiniThreadGraph.tsx`, `PlanRunRow.tsx`, `ThinkingList.tsx`, `ThreadRow.tsx`, `TodayView.tsx`, `bands.ts`, `startHere.ts`
- `src/components/today/work/` (5) : `WorkDashboard.tsx`, `dayPlan.ts`, `model.ts`, `text.ts`, `useWorkDashboard.ts`
- `src/components/ui/` (63) : `AmbientBackground.tsx`, `AnimatedCounter.tsx`, `Badge.tsx`, `Branding.tsx`, `BulkActionBar.tsx`, `Button.tsx`, `Card.tsx`, `CollapsibleMarkdown.tsx`, `CollapsibleSection.tsx`, `CompactStatCard.tsx`, `ConfirmDialog.tsx`, `Dialog.tsx`, `Dropdown.tsx`, `EmptyState.tsx`, `EntityRow.tsx`, `ErrorState.tsx`, `ExternalLink.tsx`, `FilterBar.tsx`, `FloatingMenu.tsx`, `FormDialog.tsx`, `Graph3DErrorBoundary.tsx`, `Input.tsx`, `LinkEntityDialog.tsx`, `LinkedEntityBadge.tsx`, `LoadMoreSentinel.tsx`, `MetaLine.tsx`, `MetricTooltip.tsx`, `Metrics.tsx`, `OverflowMenu.tsx`, `PageHeader.tsx`, `PageShell.tsx`, `Pagination.tsx`, `ProgressBar.tsx`, `ProgressLine.tsx`, `PulseIndicator.tsx`, `RadarChart.tsx`, `RowCheckbox.tsx`, `Section.tsx`, `SectionNav.tsx`, `Select.tsx`, `Skeleton.tsx`, `Sparkline.tsx`, `Spinner.tsx`, `StatCard.tsx`, `Status.tsx`, `StatusSelect.tsx`, `Switch.tsx`, `TabLayout.tsx`, `TaskProgress.tsx`, `Textarea.tsx`, `Toast.tsx`, `Tooltip.tsx`, `ViewTabs.tsx`, `ViewToggle.tsx`, `WatcherToggle.tsx`, `WebUpdateBanner.tsx`, `WindowedList.tsx`, `classes.ts`, `format.ts`, `index.ts`, `menuPosition.ts`, `statusMeta.ts`, `useFloatingFallback.ts`
- `src/components/universe/` (3) : `Universe3DPanel.tsx`, `index.ts`, `useEntityUniverse.ts`
- `src/constants/` (3) : `index.ts`, `intelligence.ts`, `nomenclature.ts`
- `src/hooks/` (42) : `index.ts`, `useActivationWebSocket.ts`, `useAttention.ts`, `useAttentionCount.ts`, `useBackgroundTasks.ts`, `useConfirmDialog.ts`, `useCrudEventRefresh.ts`, `useCrudEventSync.ts`, `useDetachedRuns.ts`, `useDiscussionTree.ts`, `useDragRegion.ts`, `useElapsedTime.ts`, `useEntityGroups.ts`, `useEventBus.ts`, `useFormDialog.ts`, `useIncrementalList.ts`, `useInfiniteList.ts`, `useInfiniteScroll.ts`, `useKanbanColumnData.ts`, `useKanbanFilters.ts`, `useLinkDialog.ts`, `useMediaQuery.ts`, `useMilestoneGraphData.ts`, `useModelCatalogEvents.ts`, `useMultiSelect.ts`, `usePagination.ts`, `usePipelineProgress.ts`, `usePlanGraphData.ts`, `useProjectFilter.ts`, `useSectionObserver.ts`, `useTaskGraphData.ts`, `useTaskProgress.ts`, `useToast.ts`, `useTrayNavigation.ts`, `useUpdateCheck.ts`, `useViewMode.ts`, `useViewTransition.ts`, `useVisualViewportHeight.ts`, `useVizData.ts`, `useWelcomeData.ts`, `useWindowFullscreen.ts`, `useWorkspace.ts`
- `src/hooks/runner/` (5) : `index.ts`, `useAgentExecutionsMap.ts`, `useLatestPlanRun.ts`, `useRunRootSession.ts`, `useWavesData.ts`
- `src/layouts/` (3) : `MainLayout.tsx`, `RouteErrorBoundary.tsx`, `index.ts`
- `src/lib/` (1) : `glossary.ts`
- `src/pages/` (45) : `AdminPage.tsx`, `ArchitecturePage.tsx`, `AuthCallbackPage.tsx`, `ChatSessionPage.tsx`, `CodePage.tsx`, `DecisionDetailPage.tsx`, `DecisionsPage.tsx`, `DeploymentsPage.tsx`, `DocumentsPage.tsx`, `FeatureGraphDetailPage.tsx`, `FeatureGraphsPage.tsx`, `IntelligencePage.tsx`, `LoginPage.tsx`, `McpFederationPage.tsx`, `MilestoneDetailPage.tsx`, `MilestonesPage.tsx`, `NeuralRoutingPage.tsx`, `NotFoundPage.tsx`, `NoteDetailPage.tsx`, `NotesPage.tsx`, `PersonaDetailPage.tsx`, `PersonasPage.tsx`, `PipelineDashboardPage.tsx`, `PlanDetailPage.tsx`, `PlansPage.tsx`, `ProjectDetailPage.tsx`, `ProjectMilestoneDetailPage.tsx`, `ProjectsPage.tsx`, `ProtocolDetailPage.tsx`, `ProtocolsPage.tsx`, `RfcDetailPage.tsx`, `RunnerDashboard.tsx`, `SettingsPage.tsx`, `SharingPage.tsx`, `SkillDetailPage.tsx`, `SkillsPage.tsx`, `TaskDetailPage.tsx`, `TasksPage.tsx`, `TodayPage.tsx`, `TrajectoryPage.tsx`, `TriggerDashboardPage.tsx`, `VaultPage.tsx`, `WorkspaceDetailPage.tsx`, `WorkspaceSelectorPage.tsx`, `index.ts`
- `src/pages/setup/` (7) : `AuthPage.tsx`, `ChatPage.tsx`, `InfrastructurePage.tsx`, `LaunchPage.tsx`, `SetupLayout.tsx`, `SetupWizard.tsx`, `index.ts`
- `src/services/` (35) : `admin.ts`, `api.ts`, `attention.ts`, `auth.ts`, `authManager.ts`, `code.ts`, `commits.ts`, `decisions.ts`, `discussions.ts`, `documents.ts`, `env.ts`, `environments.ts`, `eventBus.ts`, `featureGraphs.ts`, `index.ts`, `intelligence.ts`, `mcpFederation.ts`, `neighborhood.ts`, `neuralRouting.ts`, `notes.ts`, `paginate.ts`, `personas.ts`, `plans.ts`, `progress.ts`, `projects.ts`, `protocolApi.ts`, `registry.ts`, `rfcApi.ts`, `runner.ts`, `sharing.ts`, `skills.ts`, `tasks.ts`, `triggers.ts`, `vault.ts`, `workspaces.ts`
- `src/types/` (7) : `attention.ts`, `documents.ts`, `events.ts`, `fractal-graph.ts`, `index.ts`, `intelligence.ts`, `protocol.ts`
- `src/utils/` (11) : `architecture.ts`, `backgroundActivity.ts`, `chatExport.ts`, `compactYamlParser.ts`, `featureGraphModel.ts`, `featureGraphReadable.ts`, `motion.ts`, `openExternal.ts`, `paths.ts`, `stepRefreshKey.ts`, `watch.ts`
- `src/workers/` (1) : `dagreWorker.ts`

## nexus (4)

- `claude-code-api/src/bin/` (1) : `ccapi.rs`
- `claude-code-api/src/core/` (2) : `mod.rs`, `session_process.rs`
- `claude-code-sdk-rs/src/bin/` (1) : `test_interactive.rs`

## website (127)

- `src/` (5) : `App.tsx`, `Layout.tsx`, `main.tsx`, `router.tsx`, `routes.tsx`
- `src/components/` (10) : `CodeBlock.tsx`, `Counter.tsx`, `Footer.tsx`, `GithubIcon.tsx`, `Logo.tsx`, `Navbar.tsx`, `PageTransition.tsx`, `Section.tsx`, `Seo.tsx`, `ShaderBackground.tsx`
- `src/components/dashboard/` (8) : `CircularHealthGauge.tsx`, `HotspotRow.tsx`, `IconBadge.tsx`, `LayerCard.tsx`, `MetricChip.tsx`, `MiniGauge.tsx`, `healthColor.ts`, `index.ts`
- `src/components/particles/` (6) : `ParticleViz.tsx`, `fallbacks.tsx`, `index.ts`, `mocks.ts`, `useHydrated.ts`, `useParticleEngine.ts`
- `src/components/particles/adapters/` (1) : `types.ts`
- `src/components/particles/engine/` (6) : `ParticleEngine.ts`, `ParticlePool.ts`, `emitters.ts`, `forces.ts`, `index.ts`, `types.ts`
- `src/components/particles/renderer/` (2) : `CanvasRenderer.ts`, `TextRenderer.ts`
- `src/components/particles/scenes/` (17) : `AttentionScene.ts`, `ContextWindowScene.ts`, `DelegationScene.ts`, `DistributionScene.ts`, `EmbeddingsScene.ts`, `FeedbackLoopScene.ts`, `FineTuningScene.ts`, `FocusScene.ts`, `HumanAIScene.ts`, `LeverageScene.ts`, `MoatScene.ts`, `PromptOutputScene.ts`, `SignalNoiseScene.ts`, `SlideDeckScenes.ts`, `SystemScene.ts`, `index.ts`, `types.ts`
- `src/components/particles/widgets/` (7) : `CommunityVizWidget.tsx`, `ImpactPreviewWidget.tsx`, `ProjectHealthWidget.tsx`, `PropagationVizWidget.tsx`, `ProtocolRunWidget.tsx`, `WaveDispatchWidget.tsx`, `index.ts`
- `src/components/three/` (4) : `GraphCanvas.tsx`, `GraphFallback.tsx`, `GraphScene.tsx`, `graphData.ts`
- `src/components/ui/` (5) : `Badge.tsx`, `Button.tsx`, `Card.tsx`, `Container.tsx`, `index.ts`
- `src/content/` (1) : `copy.ts`
- `src/data/` (3) : `features.ts`, `pillars.ts`, `stats.ts`
- `src/hooks/` (3) : `useGsap.ts`, `useLenis.ts`, `useSsr.ts`
- `src/lib/` (8) : `demoStream.ts`, `githubApi.ts`, `iconMap.ts`, `motion.ts`, `osDetect.ts`, `seo.ts`, `splitText.tsx`, `useReducedMotion.ts`
- `src/mockups/` (4) : `DashboardMock.tsx`, `DependencyGraphMock.tsx`, `KanbanMock.tsx`, `ProductWindow.tsx`
- `src/pages/` (6) : `Features.tsx`, `Home.tsx`, `HowItWorks.tsx`, `NotFound.tsx`, `Quickstart.tsx`, `Releases.tsx`
- `src/pages/features/` (1) : `PillarSection.tsx`
- `src/providers/` (2) : `SmoothScrollProvider.tsx`, `lenisContext.ts`
- `src/sections/` (15) : `Download.tsx`, `Features.tsx`, `FeaturesBento.tsx`, `FinalCta.tsx`, `Hero.tsx`, `HeroV2.tsx`, `HeroV5.tsx`, `HowItWorksSteps.tsx`, `KnowledgeFabric.tsx`, `PlanViews.tsx`, `Playground.tsx`, `ProblemSolution.tsx`, `ScrollStory.tsx`, `SocialProof.tsx`, `Stats.tsx`
- `src/sections/playground/` (2) : `Terminal.tsx`, `scripts.ts`
- `src/state/` (4) : `index.ts`, `releaseAtom.ts`, `releasesListAtom.ts`, `uiAtoms.ts`
- `src/three/` (7) : `Edges.tsx`, `KnowledgeGraphCanvas.tsx`, `KnowledgeGraphScene.tsx`, `Nodes.tsx`, `Particles.tsx`, `StaticGraph.tsx`, `graphModel.ts`
