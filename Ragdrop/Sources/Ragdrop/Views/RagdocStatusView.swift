import SwiftUI

struct RagdocStatusView: View {
    @Bindable var store: RagdocStatusStore

    var body: some View {
        ScrollView {
        VStack(alignment: .leading, spacing: 22) {
            header
            if let error = store.errorMessage, store.snapshot != nil {
                InlineNotice(text: "Showing the last diagnostic. Refresh failed: \(error)", symbol: "exclamationmark.triangle", isError: true)
            }

            if store.isChecking && store.snapshot == nil {
                VStack(spacing: 12) {
                    ProgressView()
                    Text("Testing the MCP server and a live search…")
                        .foregroundStyle(RagdropTheme.secondary)
                    Text("This check may take around ten seconds.")
                        .font(.caption)
                        .foregroundStyle(RagdropTheme.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let snapshot = store.snapshot {
                statusContent(snapshot)
            } else if let error = store.errorMessage {
                ContentUnavailableView(
                    "Ragdoc is unreachable",
                    systemImage: "externaldrive.badge.xmark",
                    description: Text(error)
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ContentUnavailableView("Status not checked", systemImage: "externaldrive",
                                       description: Text("Run a diagnostic to check the server and search."))
            }
        }
        .padding(RagdropTheme.pagePadding)
        }
        .ragdropSurface()
        .task {
            if store.snapshot == nil { await store.refresh() }
        }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Ragdoc status")
                    .font(RagdropTheme.title)
                Text("Diagnostic of the MCP server on your backend")
                    .foregroundStyle(RagdropTheme.secondary)
            }
            Spacer()
            Button("Check again", systemImage: "arrow.clockwise") {
                Task { await store.refresh() }
            }
            .disabled(store.isChecking)
        }
    }

    private func statusContent(_ snapshot: RagdocStatusSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 14) {
                Image(systemName: snapshot.isHealthy ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                    .font(.system(size: 38))
                    .foregroundStyle(snapshot.isHealthy ? RagdropTheme.success : RagdropTheme.warning)
                VStack(alignment: .leading, spacing: 3) {
                    Text(snapshot.isHealthy ? "Ragdoc is working" : "Ragdoc needs attention")
                        .font(.title2.bold())
                    Text(snapshot.isHealthy
                         ? "The server, collection and search responded."
                         : statusExplanation(snapshot))
                        .foregroundStyle(RagdropTheme.secondary)
                }
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background((snapshot.isHealthy ? RagdropTheme.success : RagdropTheme.warning).opacity(0.10),
                        in: RoundedRectangle(cornerRadius: 14))

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                statusCard(
                    title: "MCP service",
                    symbol: "network",
                    healthy: snapshot.mcpError == nil && snapshot.listenerCount == 1 && snapshot.missingTools.isEmpty,
                    primary: snapshot.mcpError == nil ? "\(snapshot.tools.count) \(snapshot.tools.count == 1 ? "tool" : "tools") available" : "Connection failed",
                    secondary: "\(snapshot.listenerCount) network \(snapshot.listenerCount == 1 ? "listener" : "listeners") on port 8484"
                )
                statusCard(
                    title: "Canonical database",
                    symbol: "externaldrive.fill",
                    healthy: snapshot.writeState == "ready" && !snapshot.repairing,
                    primary: "\(snapshot.documents.formatted()) \(snapshot.documents == 1 ? "document" : "documents")",
                    secondary: "\(snapshot.chunks.formatted()) \(snapshot.chunks == 1 ? "passage" : "passages") · state \(snapshot.writeState)"
                )
                statusCard(
                    title: "Live search",
                    symbol: "magnifyingglass.circle.fill",
                    healthy: snapshot.searchOK && snapshot.lexicalReady && snapshot.rerankingModel == "rerank-v4.0-pro",
                    primary: snapshot.searchOK ? "Hybrid search succeeded" : "Search failed",
                    secondary: searchSummary(snapshot)
                )
                statusCard(
                    title: "Embeddings",
                    symbol: "square.stack.3d.up.fill",
                    healthy: snapshot.models[snapshot.expectedModel] == snapshot.documents,
                    primary: snapshot.expectedModel,
                    secondary: snapshot.embeddingSummary
                )
            }

            Spacer()
            HStack {
                if let revision = snapshot.revision {
                    Text("Revision \(revision.prefix(12))")
                }
                Spacer()
                if store.isChecking {
                    ProgressView().controlSize(.small)
                    Text("Checking…")
                } else if let checked = store.lastChecked {
                    Text("Checked at \(checked.formatted(date: .omitted, time: .shortened))")
                }
            }
            .font(.caption)
            .foregroundStyle(RagdropTheme.secondary)
        }
    }

    private func statusCard(
        title: String,
        symbol: String,
        healthy: Bool,
        primary: String,
        secondary: String
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label(title, systemImage: symbol)
                    .font(.headline)
                Spacer()
                Image(systemName: healthy ? "checkmark.circle.fill" : "xmark.circle.fill")
                    .foregroundStyle(healthy ? RagdropTheme.success : RagdropTheme.error)
            }
            Text(primary)
                .font(.title3.weight(.semibold))
            Text(secondary)
                .font(.caption)
                .foregroundStyle(RagdropTheme.secondary)
                .lineLimit(2)
        }
        .padding(14)
        .frame(maxWidth: .infinity, minHeight: 118, alignment: .topLeading)
        .ragdropPanel()
    }

    private func statusExplanation(_ snapshot: RagdocStatusSnapshot) -> String {
        if let error = snapshot.mcpError, !error.isEmpty { return error }
        if snapshot.listenerCount != 1 { return "The MCP network service is not available as expected." }
        if snapshot.writeState != "ready" || snapshot.repairing { return "The database is being written or repaired." }
        if !snapshot.missingTools.isEmpty { return "Required MCP tools are missing." }
        if !snapshot.searchOK { return "The server responded, but the test search failed." }
        if !snapshot.lexicalReady { return "The persistent lexical index is not ready." }
        if snapshot.rerankingModel != "rerank-v4.0-pro" { return "The expected reranking model is not active." }
        return "The diagnostic is incomplete."
    }

    private func searchSummary(_ snapshot: RagdocStatusSnapshot) -> String {
        let model = snapshot.rerankingModel ?? "unknown reranking"
        let latency = snapshot.latencySeconds.map {
            "\($0.formatted(.number.precision(.fractionLength(1)))) s"
        } ?? "unknown time"
        return "Lexical + semantic · \(model) · \(latency)"
    }
}
