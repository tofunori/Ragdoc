import SwiftUI

struct ZoteroImportView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var store: ZoteroImportStore
    @State private var query = ""
    let onAdd: ([ZoteroPDF]) -> Void

    init(store: ZoteroImportStore? = nil, isIsolated: Bool = false, excludedJobs: [ImportJob] = [], onAdd: @escaping ([ZoteroPDF]) -> Void) {
        _store = State(initialValue: store ?? ZoteroImportStore(isIsolated: isIsolated, excludedJobs: excludedJobs))
        self.onAdd = onAdd
    }

    var body: some View {
        VStack(spacing: 14) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Import from Zotero")
                        .font(.system(size: 23, weight: .semibold))
                    Text(summary)
                        .foregroundStyle(RagdropTheme.secondary)
                }
                Spacer()
                Button("Refresh", systemImage: "arrow.clockwise") {
                    Task { await store.load() }
                }
                .disabled(store.isLoading)
            }

            TextField("Search by title, author or year", text: $query)
                .textFieldStyle(.roundedBorder)

            if let error = store.errorMessage {
                ContentUnavailableView(
                    "Zotero unavailable",
                    systemImage: "books.vertical.fill",
                    description: Text(error)
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if store.isLoading && store.documents.isEmpty {
                VStack(spacing: 10) {
                    ProgressView()
                    Text("Reading Zotero and checking duplicates…")
                        .foregroundStyle(RagdropTheme.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if store.documents.isEmpty {
                ContentUnavailableView("No local PDFs available", systemImage: "books.vertical",
                                       description: Text("Local Zotero PDF attachments will appear here."))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if filteredDocuments.isEmpty {
                ContentUnavailableView("No articles found", systemImage: "magnifyingglass", description: Text("Try another title, author or year."))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List(filteredDocuments) { document in
                    HStack(spacing: 12) {
                        Toggle("", isOn: Binding(
                            get: { store.selectedKeys.contains(document.attachmentKey) },
                            set: { store.setSelected($0, key: document.attachmentKey) }
                        ))
                        .labelsHidden()
                        .accessibilityLabel("Select \(document.title)")
                        .disabled(document.isIndexed)

                        VStack(alignment: .leading, spacing: 3) {
                            Text(document.title)
                                .fontWeight(.medium)
                                .lineLimit(1)
                            Text(metadata(for: document))
                                .font(.caption)
                                .foregroundStyle(RagdropTheme.secondary)
                                .lineLimit(1)
                        }
                        Spacer()
                        if document.isIndexed {
                            Label("Already indexed", systemImage: "checkmark.circle.fill")
                                .font(.caption.weight(.medium))
                                .foregroundStyle(RagdropTheme.success)
                        } else {
                            Text("To add")
                                .font(.caption.weight(.medium))
                                .foregroundStyle(RagdropTheme.secondary)
                        }
                    }
                    .padding(.vertical, 3)
                }
                .listStyle(.inset)
                .scrollContentBackground(.hidden)
                .background(RagdropTheme.panel)
                .clipShape(.rect(cornerRadius: 12))
            }

            Divider()
            HStack {
                Button("Select all") { store.select(filteredDocuments) }
                    .disabled(filteredDocuments.allSatisfy(\.isIndexed))
                Button("Clear selection", action: store.clearSelection)
                    .disabled(store.selectedKeys.isEmpty)
                Spacer()
                Button("Close", role: .cancel) { dismiss() }
                Button(addButtonTitle) {
                    onAdd(store.selectedDocuments)
                    dismiss()
                }
                .buttonStyle(RagdropPrimaryButtonStyle())
                .disabled(store.selectedKeys.isEmpty || store.isLoading || store.errorMessage != nil)
            }
        }
        .padding(RagdropTheme.pagePadding)
        .frame(minWidth: 760, idealWidth: 880, minHeight: 580, idealHeight: 680)
        .ragdropSurface()
        .task { await store.load() }
    }

    private var filteredDocuments: [ZoteroPDF] {
        let needle = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !needle.isEmpty else { return store.documents }
        return store.documents.filter {
            $0.title.localizedCaseInsensitiveContains(needle)
                || $0.authors.localizedCaseInsensitiveContains(needle)
                || ($0.year?.localizedCaseInsensitiveContains(needle) ?? false)
        }
    }

    private var summary: String {
        guard !store.documents.isEmpty else { return "Select articles to add to Ragdoc" }
        let indexed = store.documents.filter(\.isIndexed).count
        return "\(RagdropText.pdfCount(store.documents.count)) available locally · \(indexed) already indexed"
    }

    private var addButtonTitle: String {
        let count = store.selectedKeys.count
        return count == 1 ? "Add 1 PDF to queue" : "Add \(count) PDFs to queue"
    }

    private func metadata(for document: ZoteroPDF) -> String {
        [document.authors.nonBlank, document.year, document.fileURL.lastPathComponent]
            .compactMap { $0 }
            .joined(separator: " · ")
    }
}
