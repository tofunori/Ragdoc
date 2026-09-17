import SwiftUI

struct RagdocStatusView: View {
    @Bindable var store: RagdocStatusStore

    var body: some View {
        ScrollView {
        VStack(alignment: .leading, spacing: 22) {
            header
            if let error = store.errorMessage, store.snapshot != nil {
                InlineNotice(text: "Dernier diagnostic conservé. Actualisation échouée : \(error)", symbol: "exclamationmark.triangle", isError: true)
            }

            if store.isChecking && store.snapshot == nil {
                VStack(spacing: 12) {
                    ProgressView()
                    Text("Test du serveur MCP et d’une recherche réelle…")
                        .foregroundStyle(RagdropTheme.secondary)
                    Text("Cette vérification peut prendre une dizaine de secondes.")
                        .font(.caption)
                        .foregroundStyle(RagdropTheme.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let snapshot = store.snapshot {
                statusContent(snapshot)
            } else if let error = store.errorMessage {
                ContentUnavailableView(
                    "Ragdoc est inaccessible",
                    systemImage: "externaldrive.badge.xmark",
                    description: Text(error)
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ContentUnavailableView("État non vérifié", systemImage: "externaldrive",
                                       description: Text("Lancez un diagnostic pour vérifier le serveur et la recherche."))
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
                Text("État de Ragdoc")
                    .font(RagdropTheme.title)
                Text("Diagnostic du MCP actif sur le NAS")
                    .foregroundStyle(RagdropTheme.secondary)
            }
            Spacer()
            Button("Revérifier", systemImage: "arrow.clockwise") {
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
                    Text(snapshot.isHealthy ? "Ragdoc fonctionne correctement" : "Ragdoc demande votre attention")
                        .font(.title2.bold())
                    Text(snapshot.isHealthy
                         ? "Le serveur, la collection et la recherche ont répondu."
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
                    title: "Service MCP",
                    symbol: "network",
                    healthy: snapshot.mcpError == nil && snapshot.listenerCount == 1 && snapshot.missingTools.isEmpty,
                    primary: snapshot.mcpError == nil ? "\(snapshot.tools.count) outils disponibles" : "Connexion échouée",
                    secondary: "\(snapshot.listenerCount) service réseau sur le port 8484"
                )
                statusCard(
                    title: "Base canonique",
                    symbol: "externaldrive.fill",
                    healthy: snapshot.writeState == "ready" && !snapshot.repairing,
                    primary: "\(snapshot.documents.formatted()) documents",
                    secondary: "\(snapshot.chunks.formatted()) passages · état \(snapshot.writeState)"
                )
                statusCard(
                    title: "Recherche réelle",
                    symbol: "magnifyingglass.circle.fill",
                    healthy: snapshot.searchOK && snapshot.lexicalReady && snapshot.rerankingModel == "rerank-v4.0-pro",
                    primary: snapshot.searchOK ? "Recherche hybride réussie" : "Recherche échouée",
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
                    Text("Révision \(revision.prefix(12))")
                }
                Spacer()
                if store.isChecking {
                    ProgressView().controlSize(.small)
                    Text("Vérification…")
                } else if let checked = store.lastChecked {
                    Text("Vérifié à \(checked.formatted(date: .omitted, time: .shortened))")
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
        if snapshot.listenerCount != 1 { return "Le service réseau MCP n’est pas disponible normalement." }
        if snapshot.writeState != "ready" || snapshot.repairing { return "La base est en cours d’écriture ou de réparation." }
        if !snapshot.missingTools.isEmpty { return "Des outils MCP requis sont absents." }
        if !snapshot.searchOK { return "Le serveur répond, mais la recherche test a échoué." }
        if !snapshot.lexicalReady { return "L’index lexical persistant n’est pas prêt." }
        if snapshot.rerankingModel != "rerank-v4.0-pro" { return "Le modèle de reranking attendu n’est pas actif." }
        return "Le diagnostic est incomplet."
    }

    private func searchSummary(_ snapshot: RagdocStatusSnapshot) -> String {
        let model = snapshot.rerankingModel ?? "reranking inconnu"
        let latency = snapshot.latencySeconds.map {
            "\($0.formatted(.number.precision(.fractionLength(1)))) s"
        } ?? "temps inconnu"
        return "Lexical + sémantique · \(model) · \(latency)"
    }
}
