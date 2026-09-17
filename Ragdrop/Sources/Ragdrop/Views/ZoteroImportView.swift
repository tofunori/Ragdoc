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
                    Text("Importer depuis Zotero")
                        .font(.system(size: 23, weight: .semibold))
                    Text(summary)
                        .foregroundStyle(RagdropTheme.secondary)
                }
                Spacer()
                Button("Actualiser", systemImage: "arrow.clockwise") {
                    Task { await store.load() }
                }
                .disabled(store.isLoading)
            }

            TextField("Rechercher un titre, un auteur ou une année", text: $query)
                .textFieldStyle(.roundedBorder)

            if let error = store.errorMessage {
                ContentUnavailableView(
                    "Zotero indisponible",
                    systemImage: "books.vertical.fill",
                    description: Text(error)
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if store.isLoading && store.documents.isEmpty {
                VStack(spacing: 10) {
                    ProgressView()
                    Text("Lecture de Zotero et vérification des doublons…")
                        .foregroundStyle(RagdropTheme.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if store.documents.isEmpty {
                ContentUnavailableView("Aucun PDF local disponible", systemImage: "books.vertical",
                                       description: Text("Les pièces jointes PDF locales de Zotero seront proposées ici."))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if filteredDocuments.isEmpty {
                ContentUnavailableView("Aucun article trouvé", systemImage: "magnifyingglass", description: Text("Essayez un autre titre, un auteur ou une année."))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List(filteredDocuments) { document in
                    HStack(spacing: 12) {
                        Toggle("", isOn: Binding(
                            get: { store.selectedKeys.contains(document.attachmentKey) },
                            set: { store.setSelected($0, key: document.attachmentKey) }
                        ))
                        .labelsHidden()
                        .accessibilityLabel("Sélectionner \(document.title)")
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
                            Label("Déjà indexé", systemImage: "checkmark.circle.fill")
                                .font(.caption.weight(.medium))
                                .foregroundStyle(RagdropTheme.success)
                        } else {
                            Text("À ajouter")
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
                Button("Tout sélectionner") { store.select(filteredDocuments) }
                    .disabled(filteredDocuments.allSatisfy(\.isIndexed))
                Button("Effacer la sélection", action: store.clearSelection)
                    .disabled(store.selectedKeys.isEmpty)
                Spacer()
                Button("Fermer", role: .cancel) { dismiss() }
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
        guard !store.documents.isEmpty else { return "Sélectionnez les articles à ajouter à Ragdoc" }
        let indexed = store.documents.filter(\.isIndexed).count
        return "\(store.documents.count) PDF \(store.documents.count == 1 ? "local" : "locaux") · \(indexed) déjà indexé\(indexed > 1 ? "s" : "")"
    }

    private var addButtonTitle: String {
        let count = store.selectedKeys.count
        return count == 1 ? "Ajouter 1 PDF à la file" : "Ajouter \(count) PDF à la file"
    }

    private func metadata(for document: ZoteroPDF) -> String {
        [document.authors.nonBlank, document.year, document.fileURL.lastPathComponent]
            .compactMap { $0 }
            .joined(separator: " · ")
    }
}
