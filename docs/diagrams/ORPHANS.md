<!-- Genere par scripts/diagrams/check-index.mjs --write-orphans. Ne pas editer a la main. -->

# Fichiers source sans diagramme proprietaire

Un fichier source est **orphelin** quand aucun `covers` de `INDEX.yml` ne le matche :
aucun diagramme ne repond de son comportement. Cette liste est publiee pour etre honnete
sur ce que la cartographie couvre reellement — on ne reduit pas le denominateur, on la reduit elle.

**734 orphelins sur 1204 fichiers source (61.0 %).**

Regeneration (hors reseau) :

```
node scripts/diagrams/check-index.mjs --write-orphans
```

## backend (162)

- `src/analytics/stats/` (4) : `anova.rs`, `fitting.rs`, `golden_fixtures.rs`, `mean_std.rs`
- `src/api/` (21) : `chat_handlers.rs`, `code_handlers.rs`, `embedded_frontend.rs`, `environment_handlers.rs`, `episode_handlers.rs`, `feedback_handlers.rs`, `graph_handlers.rs`, `graph_types.rs`, `neural_routing_handlers.rs`, `note_handlers.rs`, `persona_handlers.rs`, `profile_handlers.rs`, `protocol_handlers.rs`, `reason_handlers.rs`, `registry_handlers.rs`, `rfc_handlers.rs`, `skill_handlers.rs`, `trajectory_handlers.rs`, `trigger_handlers.rs`, `vault_handlers.rs`, `ws_run_handler.rs`
- `src/auth/` (5) : `google.rs`, `jwt.rs`, `oauth_server.rs`, `oidc.rs`, `refresh.rs`
- `src/chat/` (19) : `cli_auth.rs`, `cli_version.rs`, `compaction_context.rs`, `composer.rs`, `continuity.rs`, `enrichment.rs`, `entity_extractor.rs`, `feedback.rs`, `hook_ledger.rs`, `model_catalog.rs`, `observation_detector.rs`, `path_detect.rs`, `post_tool_hook.rs`, `prompt.rs`, `prompt_sections.rs`, `routing.rs`, `skill_hook.rs`, `viz.rs`, `viz_builder.rs`
- `src/chat/stages/` (9) : `biomimicry.rs`, `file_context.rs`, `intent_weights.rs`, `knowledge_injection.rs`, `mod.rs`, `persona.rs`, `skill_activation.rs`, `status_injection.rs`, `user_profile.rs`
- `src/episodes/` (3) : `artifact_comparison.rs`, `distill_models.rs`, `evaluation.rs`
- `src/events/` (4) : `graph.rs`, `nats.rs`, `notifier.rs`, `trigger_routing.rs`
- `src/graph/` (2) : `mock.rs`, `neighborhood.rs`
- `src/neo4j/` (24) : `agent_execution.rs`, `alert.rs`, `chat.rs`, `code.rs`, `commit.rs`, `constraint.rs`, `decision.rs`, `environment.rs`, `event_trigger.rs`, `feature_graph.rs`, `mcp_federation.rs`, `mock.rs`, `models.rs`, `neighborhood.rs`, `persona.rs`, `plan_run.rs`, `profile.rs`, `protocol.rs`, `reasoning.rs`, `sharing.rs`, `skill.rs`, `topology.rs`, `trigger.rs`, `user.rs`
- `src/orchestrator/` (3) : `context.rs`, `planner.rs`, `topology_hook.rs`
- `src/parser/languages/` (18) : `bash.rs`, `c.rs`, `cpp.rs`, `csharp.rs`, `dart.rs`, `go.rs`, `hcl.rs`, `java.rs`, `kotlin.rs`, `mod.rs`, `php.rs`, `python.rs`, `ruby.rs`, `rust.rs`, `scala.rs`, `swift.rs`, `typescript.rs`, `zig.rs`
- `src/pipeline/` (14) : `composer.rs`, `critic.rs`, `episode_adapter.rs`, `evolve.rs`, `feedback.rs`, `gates.rs`, `materialize.rs`, `metrics.rs`, `mod.rs`, `progress.rs`, `regression.rs`, `runner.rs`, `skill_injector.rs`, `wave_executor.rs`
- `src/runner/` (10) : `enricher.rs`, `feedback.rs`, `feedback_analyzer.rs`, `git.rs`, `lifecycle.rs`, `persona.rs`, `prompt.rs`, `trigger.rs`, `vector.rs`, `verifier.rs`
- `src/runner/providers/` (4) : `event.rs`, `mod.rs`, `schedule.rs`, `webhook.rs`
- `src/skills/` (12) : `cache.rs`, `export.rs`, `hook_extractor.rs`, `import.rs`, `naming.rs`, `package.rs`, `project_resolver.rs`, `registry.rs`, `templates.rs`, `triggers.rs`, `trust.rs`, `validation.rs`
- `src/utils/` (3) : `file_path_extractor.rs`, `mod.rs`, `paths.rs`
- `src/vault/` (7) : `agent_cli.rs`, `crypto.rs`, `grants.rs`, `mask.rs`, `mod.rs`, `service.rs`, `store.rs`

## frontend (505)

- `src/adapters/` (3) : `MilestoneGraphAdapter.ts`, `PlanGraphAdapter.ts`, `TaskGraphAdapter.ts`
- `src/atoms/` (12) : `auth.ts`, `events.ts`, `index.ts`, `intelligence.ts`, `modelCatalog.ts`, `notes.ts`, `plans.ts`, `projects.ts`, `setup.ts`, `tasks.ts`, `ui.ts`, `workspaces.ts`
- `src/components/` (4) : `DependencyGraphView.tsx`, `TitleBar.tsx`, `UpdateBanner.tsx`, `WorkspaceSwitcher.tsx`
- `src/components/auth/` (3) : `PasswordLoginForm.tsx`, `RegisterForm.tsx`, `UserMenu.tsx`
- `src/components/chat/` (35) : `AgentGroup.tsx`, `AgenticModeBanner.tsx`, `AgenticModePill.tsx`, `AskUserQuestionBlock.tsx`, `Attachments.tsx`, `BackgroundActivityBlock.tsx`, `BackgroundTasksIndicator.tsx`, `ChatInput.tsx`, `ChatMessages.tsx`, `ChatPanel.tsx`, `ChatWelcome.tsx`, `CompactBoundaryBlock.tsx`, `CompactionBanner.tsx`, `ContinueIndicatorBlock.tsx`, `CopyMarkdownButton.tsx`, `DetachedRunsPanel.tsx`, `InputRequestBlock.tsx`, `MarkdownText.tsx`, `MessageQueueBar.tsx`, `ModelChangedBlock.tsx`, `ModelFamilyPicker.tsx`, `PermissionSettingsPanel.tsx`, `ProjectSelect.tsx`, `ResultErrorBlock.tsx`, `ResultMaxTurnsBlock.tsx`, `RetryIndicatorBlock.tsx`, `SessionBreadcrumb.tsx`, `SystemHintBlock.tsx`, `SystemInitBlock.tsx`, `ThinkingBlock.tsx`, `ToolCallBlock.tsx`, `ToolCallGroup.tsx`, `attachmentState.ts`, `index.ts`, `useElapsedMs.ts`
- `src/components/chat/tools/` (13) : `BashToolRenderer.tsx`, `DefaultToolRenderer.tsx`, `EditToolRenderer.tsx`, `McpToolRenderer.tsx`, `ReadToolRenderer.tsx`, `SearchToolRenderer.tsx`, `TodoWriteRenderer.tsx`, `WebToolRenderer.tsx`, `WriteToolRenderer.tsx`, `index.tsx`, `summaries.ts`, `syntax.ts`, `types.ts`
- `src/components/chat/tools/mcp/` (7) : `ChatRenderer.tsx`, `CodeRenderer.tsx`, `EntityRenderer.tsx`, `ListRenderer.tsx`, `ProgressRenderer.tsx`, `index.tsx`, `utils.tsx`
- `src/components/chat/viz/` (10) : `ContextRadarViz.tsx`, `ImpactGraphViz.tsx`, `KnowledgeCardViz.tsx`, `ProgressBarViz.tsx`, `ProtocolRunViz.tsx`, `ReasoningTreeViz.tsx`, `VizBlockRenderer.tsx`, `VizExpandDialog.tsx`, `index.ts`, `registry.ts`
- `src/components/code/` (11) : `CoChangeGraph.tsx`, `CodeArchitectureFullTab.tsx`, `CodeArchitectureTab.tsx`, `CodeCommunitiesTab.tsx`, `CodeExplorerTab.tsx`, `CodeHealthTab.tsx`, `CodeHeritageTab.tsx`, `CodeProcessesTab.tsx`, `CodeSanteTab.tsx`, `FileHistoryDrawer.tsx`, `index.ts`
- `src/components/commits/` (2) : `CommitList.tsx`, `index.ts`
- `src/components/composer/` (6) : `FSMCanvas.tsx`, `NotePool.tsx`, `PatternComposer.tsx`, `TriggerBuilder.tsx`, `index.ts`, `types.ts`
- `src/components/discussions/` (3) : `DiscussionNode.tsx`, `DiscussionTreeView.tsx`, `InlineConversationPanel.tsx`
- `src/components/expandable/` (1) : `index.tsx`
- `src/components/forms/` (26) : `AutoBuildFeatureGraphForm.tsx`, `CreateComponentForm.tsx`, `CreateConstraintForm.tsx`, `CreateDecisionForm.tsx`, `CreateFeatureGraphForm.tsx`, `CreateMilestoneForm.tsx`, `CreateNoteForm.tsx`, `CreatePlanForm.tsx`, `CreateProjectForm.tsx`, `CreateReleaseForm.tsx`, `CreateResourceForm.tsx`, `CreateSkillForm.tsx`, `CreateStepForm.tsx`, `CreateTaskForm.tsx`, `CreateWorkspaceForm.tsx`, `DecisionForms.tsx`, `EditMilestoneForm.tsx`, `EditPersonaForm.tsx`, `EditPlanForm.tsx`, `EditProjectForm.tsx`, `EditStepForm.tsx`, `EditTaskForm.tsx`, `EditWorkspaceForm.tsx`, `ImportSkillForm.tsx`, `NoteForms.tsx`, `index.ts`
- `src/components/graph/` (1) : `EntityGroupPanel.tsx`
- `src/components/graph/entity/` (11) : `EntityGraphCanvas.tsx`, `EntityGraphControls.tsx`, `EntityGraphExplainer.tsx`, `EntityTypeIcon.tsx`, `NodeInfoCard.tsx`, `entityHref.ts`, `entityVisuals.ts`, `index.ts`, `radialLayout.ts`, `usePanZoom.ts`, `useReducedMotion.ts`
- `src/components/intelligence/` (22) : `ActivityHeatmap3D.tsx`, `ContextRadar.tsx`, `GraphLoadingProgress.tsx`, `IntelligenceDashboard.tsx`, `IntelligenceGraphPage.tsx`, `LayerControls.tsx`, `LearningTimeline.tsx`, `LiveIndicator.tsx`, `NodeInspector.tsx`, `ProtocolRanking.tsx`, `ProtocolRunViewer.tsx`, `SpreadingActivation.tsx`, `VectorSpaceExplorer.tsx`, `WorkspaceGraphPage.tsx`, `WorkspaceLearningTimeline.tsx`, `index.ts`, `useGraphWebSocket.ts`, `useIntelligenceGraph.ts`, `useProtocolRunEvents.ts`, `useWorkspaceIntelligenceData.ts`, `useWorkspaceIntelligenceGraph.ts`, `useWsAnimation.ts`
- `src/components/intelligence/cards/` (4) : `FileContextCard.tsx`, `NoteContextCard.tsx`, `ProtocolContextCard.tsx`, `SkillContextCard.tsx`
- `src/components/intelligence/edges/` (4) : `AffectsEdge.tsx`, `CoChangedEdge.tsx`, `SynapseEdge.tsx`, `index.ts`
- `src/components/intelligence/graph3d/` (6) : `CommunityHulls3D.tsx`, `IntelligenceGraph3D.tsx`, `nodeObjects.ts`, `useActivationSync.ts`, `useGraph3DLayout.ts`, `useRenderLoop.ts`
- `src/components/intelligence/nodes/` (14) : `DecisionNode.tsx`, `EnumNode.tsx`, `FeatureGraphNode.tsx`, `FileNode.tsx`, `FunctionNode.tsx`, `NoteNode.tsx`, `PlanNode.tsx`, `ProtocolNode.tsx`, `ProtocolStateNode.tsx`, `SkillNode.tsx`, `StructNode.tsx`, `TaskNode.tsx`, `TraitNode.tsx`, `index.ts`
- `src/components/intelligence/vectorspace3d/` (1) : `VectorSpace3D.tsx`
- `src/components/kanban/` (10) : `BoardCard.tsx`, `KanbanCard.tsx`, `KanbanFilterBar.tsx`, `ListControls.tsx`, `MilestoneKanbanCard.tsx`, `PlanKanbanCard.tsx`, `PlanKanbanFilterBar.tsx`, `UniversalKanbanCard.tsx`, `UniversalKanbanColumn.tsx`, `index.ts`
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
- `src/components/protocols/` (10) : `Explainer.tsx`, `FsmBreadcrumbs.tsx`, `GanttTimeline.tsx`, `RecentRunsPanel.tsx`, `RfcDashboardPage.tsx`, `RfcStatusBadge.tsx`, `RunStatusBadge.tsx`, `ScheduledActionsPanel.tsx`, `index.ts`, `runHelpers.ts`
- `src/components/registry/` (7) : `ImportWizard.tsx`, `SkillBrowser.tsx`, `TrustBadge.tsx`, `concepts.tsx`, `fetchAll.ts`, `index.ts`, `metrics.ts`
- `src/components/runner/` (15) : `AgentExecutionDetail.tsx`, `BudgetEditor.tsx`, `CancelButton.tsx`, `ConversationPanel.tsx`, `InlineConversation.tsx`, `LiveProgress.tsx`, `PlanRunHistory.tsx`, `PlanRunRow.tsx`, `RunnerHeader.tsx`, `StatsRow.tsx`, `WaveAgentCard.tsx`, `WaveSection.tsx`, `WsStatusIndicator.tsx`, `index.ts`, `shared.ts`
- `src/components/settings/` (1) : `SettingRow.tsx`
- `src/components/tasks/` (5) : `DetailRows.tsx`, `RowStateLink.tsx`, `StatusBreakdown.tsx`, `TaskUniverse3D.tsx`, `useTaskUniverse.ts`
- `src/components/ui/` (62) : `AmbientBackground.tsx`, `AnimatedCounter.tsx`, `Badge.tsx`, `Branding.tsx`, `BulkActionBar.tsx`, `Button.tsx`, `Card.tsx`, `CollapsibleMarkdown.tsx`, `CollapsibleSection.tsx`, `CompactStatCard.tsx`, `ConfirmDialog.tsx`, `Dialog.tsx`, `Dropdown.tsx`, `EmptyState.tsx`, `EntityRow.tsx`, `ErrorState.tsx`, `ExternalLink.tsx`, `FilterBar.tsx`, `FloatingMenu.tsx`, `FormDialog.tsx`, `Graph3DErrorBoundary.tsx`, `Input.tsx`, `LinkEntityDialog.tsx`, `LinkedEntityBadge.tsx`, `LoadMoreSentinel.tsx`, `MetaLine.tsx`, `MetricTooltip.tsx`, `Metrics.tsx`, `OverflowMenu.tsx`, `PageHeader.tsx`, `PageShell.tsx`, `Pagination.tsx`, `ProgressBar.tsx`, `ProgressLine.tsx`, `PulseIndicator.tsx`, `RadarChart.tsx`, `RowCheckbox.tsx`, `Section.tsx`, `SectionNav.tsx`, `Select.tsx`, `Skeleton.tsx`, `Sparkline.tsx`, `Spinner.tsx`, `StatCard.tsx`, `Status.tsx`, `StatusSelect.tsx`, `Switch.tsx`, `TabLayout.tsx`, `TaskProgress.tsx`, `Textarea.tsx`, `Toast.tsx`, `Tooltip.tsx`, `ViewTabs.tsx`, `ViewToggle.tsx`, `WatcherToggle.tsx`, `WebUpdateBanner.tsx`, `classes.ts`, `format.ts`, `index.ts`, `menuPosition.ts`, `statusMeta.ts`, `useFloatingFallback.ts`
- `src/components/universe/` (3) : `Universe3DPanel.tsx`, `index.ts`, `useEntityUniverse.ts`
- `src/constants/` (3) : `index.ts`, `intelligence.ts`, `models.ts`
- `src/hooks/` (35) : `index.ts`, `useActivationWebSocket.ts`, `useBackgroundTasks.ts`, `useChatUrlSync.ts`, `useConfirmDialog.ts`, `useDetachedRuns.ts`, `useDiscussionTree.ts`, `useDragRegion.ts`, `useElapsedTime.ts`, `useEntityGroups.ts`, `useEventBus.ts`, `useFormDialog.ts`, `useInfiniteList.ts`, `useInfiniteScroll.ts`, `useKanbanFilters.ts`, `useLinkDialog.ts`, `useMediaQuery.ts`, `useMilestoneGraphData.ts`, `useModelCatalogEvents.ts`, `useMultiSelect.ts`, `usePagination.ts`, `usePipelineProgress.ts`, `useProjectFilter.ts`, `useSectionObserver.ts`, `useTaskGraphData.ts`, `useTaskProgress.ts`, `useToast.ts`, `useTrayNavigation.ts`, `useUpdateCheck.ts`, `useViewTransition.ts`, `useVisualViewportHeight.ts`, `useVizData.ts`, `useWelcomeData.ts`, `useWindowFullscreen.ts`, `useWorkspace.ts`
- `src/hooks/runner/` (6) : `index.ts`, `useAgentExecutionsMap.ts`, `useConversationWs.ts`, `useLatestPlanRun.ts`, `useRunRootSession.ts`, `useWavesData.ts`
- `src/lib/` (1) : `glossary.ts`
- `src/pages/` (38) : `AdminPage.tsx`, `ArchitecturePage.tsx`, `AuthCallbackPage.tsx`, `ChatSessionPage.tsx`, `CodePage.tsx`, `DecisionDetailPage.tsx`, `DecisionsPage.tsx`, `DeploymentsPage.tsx`, `DocumentsPage.tsx`, `FeatureGraphDetailPage.tsx`, `FeatureGraphsPage.tsx`, `IntelligencePage.tsx`, `LoginPage.tsx`, `McpFederationPage.tsx`, `MilestoneDetailPage.tsx`, `MilestonesPage.tsx`, `NeuralRoutingPage.tsx`, `NotFoundPage.tsx`, `NoteDetailPage.tsx`, `PersonaDetailPage.tsx`, `PersonasPage.tsx`, `PipelineDashboardPage.tsx`, `PlanDetailPage.tsx`, `ProjectDetailPage.tsx`, `ProjectMilestoneDetailPage.tsx`, `ProjectsPage.tsx`, `RfcDetailPage.tsx`, `RunnerDashboard.tsx`, `SettingsPage.tsx`, `SharingPage.tsx`, `SkillDetailPage.tsx`, `SkillsPage.tsx`, `TaskDetailPage.tsx`, `TrajectoryPage.tsx`, `TriggerDashboardPage.tsx`, `WorkspaceDetailPage.tsx`, `WorkspaceSelectorPage.tsx`, `index.ts`
- `src/pages/setup/` (7) : `AuthPage.tsx`, `ChatPage.tsx`, `InfrastructurePage.tsx`, `LaunchPage.tsx`, `SetupLayout.tsx`, `SetupWizard.tsx`, `index.ts`
- `src/services/` (34) : `admin.ts`, `api.ts`, `auth.ts`, `authManager.ts`, `chat.ts`, `code.ts`, `commits.ts`, `decisions.ts`, `discussions.ts`, `documents.ts`, `env.ts`, `environments.ts`, `featureGraphs.ts`, `index.ts`, `intelligence.ts`, `mcpFederation.ts`, `neighborhood.ts`, `neuralRouting.ts`, `notes.ts`, `paginate.ts`, `personas.ts`, `plans.ts`, `progress.ts`, `projects.ts`, `protocolApi.ts`, `registry.ts`, `rfcApi.ts`, `runner.ts`, `sharing.ts`, `skills.ts`, `tasks.ts`, `triggers.ts`, `workspaces.ts`, `wsAdapter.ts`
- `src/types/` (7) : `chat.ts`, `documents.ts`, `events.ts`, `fractal-graph.ts`, `index.ts`, `intelligence.ts`, `protocol.ts`
- `src/utils/` (7) : `architecture.ts`, `chatExport.ts`, `compactYamlParser.ts`, `motion.ts`, `openExternal.ts`, `paths.ts`, `watch.ts`
- `src/workers/` (1) : `dagreWorker.ts`

## nexus (67)

- `claude-code-api/src/` (1) : `main.rs`
- `claude-code-api/src/api/` (8) : `chat.rs`, `conversations.rs`, `mod.rs`, `models.rs`, `projects.rs`, `sessions.rs`, `stats.rs`, `streaming_handler.rs`
- `claude-code-api/src/bin/` (1) : `ccapi.rs`
- `claude-code-api/src/core/` (13) : `auth.rs`, `cache.rs`, `claude_manager.rs`, `config.rs`, `conversation.rs`, `interactive_session.rs`, `mod.rs`, `model_registry.rs`, `objective_tracker.rs`, `process_pool.rs`, `retry.rs`, `session_manager.rs`, `session_process.rs`
- `claude-code-api/src/core/hooks/` (3) : `mod.rs`, `neo4j_hook_callback.rs`, `neo4j_permission_provider.rs`
- `claude-code-api/src/core/memory/` (6) : `long_term.rs`, `medium_term.rs`, `mod.rs`, `short_term.rs`, `traits.rs`, `unified.rs`
- `claude-code-api/src/core/storage/` (7) : `combined.rs`, `meilisearch.rs`, `memory.rs`, `mod.rs`, `neo4j.rs`, `tiered_cache.rs`, `traits.rs`
- `claude-code-api/src/middleware/` (3) : `error_handler.rs`, `mod.rs`, `request_id.rs`
- `claude-code-api/src/models/` (4) : `claude.rs`, `error.rs`, `mod.rs`, `openai.rs`
- `claude-code-api/src/utils/` (5) : `function_calling.rs`, `mod.rs`, `parser.rs`, `streaming.rs`, `text_chunker.rs`
- `claude-code-sdk-rs/src/` (8) : `cli_download.rs`, `client_working.rs`, `errors.rs`, `model_recommendation.rs`, `optimized_client.rs`, `perf_utils.rs`, `sdk_mcp.rs`, `token_tracker.rs`
- `claude-code-sdk-rs/src/bin/` (1) : `test_interactive.rs`
- `claude-code-sdk-rs/src/memory/` (6) : `integration.rs`, `message_document.rs`, `mod.rs`, `provider.rs`, `scoring.rs`, `tool_context.rs`
- `claude-code-sdk-rs/src/transport/` (1) : `mock.rs`
