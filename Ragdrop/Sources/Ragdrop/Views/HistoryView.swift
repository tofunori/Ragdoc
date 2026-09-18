import SwiftUI

struct HistoryView: View {
    @Bindable var store: HistoryStore
    @Binding var query: String

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header


            if let error = store.errorMessage {
                errorBanner(error)
            }

            Group {
                if store.isLoading && store.documents.isEmpty {
                    VStack(spacing: 12) {
                        ProgressView()
                        Text("Reading the Ragdoc library…")
                            .foregroundStyle(RagdropTheme.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if store.errorMessage != nil, store.documents.isEmpty {
                    ContentUnavailableView(
                        "Library unavailable",
                        systemImage: "externaldrive.badge.exclamationmark",
                        description: Text("Check the server connection, then refresh.")
                    )
                } else if store.documents.isEmpty {
                    ContentUnavailableView("Your library is empty", systemImage: "books.vertical",
                                           description: Text("Articles added to Ragdoc will appear here."))
                } else if filteredDocuments.isEmpty {
                    ContentUnavailableView("No articles found", systemImage: "magnifyingglass", description: Text("Try another title, source or DOI."))
                } else {
                    documentTable
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .padding(RagdropTheme.pagePadding)
        .ragdropSurface()
        .task {
            if store.documents.isEmpty { await store.refresh() }
        }
    }

    private func errorBanner(_ message: String) -> some View {
        Label(message, systemImage: "exclamationmark.triangle.fill")
            .font(.callout)
            .foregroundStyle(RagdropTheme.error)
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RagdropTheme.error.opacity(0.09), in: RoundedRectangle(cornerRadius: 10))
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Library")
                    .font(RagdropTheme.title)
                Text("\(filteredDocuments.count) / \(store.documents.count) documents in the canonical database")
                    .foregroundStyle(RagdropTheme.secondary)
            }
            Spacer()
            Button("Refresh", systemImage: "arrow.clockwise") { Task { await store.refresh() } }
                .disabled(store.isLoading)
            if store.isLoading {
                ProgressView().controlSize(.small)
            } else if let lastUpdated = store.lastUpdated {
                Text("Updated at \(lastUpdated.formatted(date: .omitted, time: .shortened))")
                    .font(.caption)
                    .foregroundStyle(RagdropTheme.secondary)
            }
        }
    }

    private var documentTable: some View {
        Table(filteredDocuments) {
            TableColumn("Article") { document in
                VStack(alignment: .leading, spacing: 2) {
                    Text(document.displayTitle)
                        .lineLimit(1)
                    Text(document.source)
                        .font(.caption)
                        .foregroundStyle(RagdropTheme.secondary)
                        .lineLimit(1)
                }
                .padding(.vertical, 3)
            }
            .width(min: 330, ideal: 480)

            TableColumn("Passages") { document in
                Text(document.chunks.formatted())
                    .monospacedDigit()
            }
            .width(min: 70, ideal: 85, max: 95)

            TableColumn("Indexed on") { document in
                Text(document.displayDate)
                    .foregroundStyle(RagdropTheme.secondary)
                    .monospacedDigit()
            }
            .width(min: 125, ideal: 145, max: 155)
        }
        .tableStyle(.inset)
        .alternatingRowBackgrounds(.disabled)
        .scrollContentBackground(.hidden)
        .background(RagdropTheme.panel)
        .clipShape(.rect(cornerRadius: RagdropTheme.radius))
    }

    private var filteredDocuments: [HistoryDocument] {
        let needle = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !needle.isEmpty else { return store.documents }
        return store.documents.filter {
            $0.displayTitle.localizedCaseInsensitiveContains(needle)
                || $0.source.localizedCaseInsensitiveContains(needle)
                || ($0.doi?.localizedCaseInsensitiveContains(needle) ?? false)
        }
    }
}
