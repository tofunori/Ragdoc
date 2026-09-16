import Foundation

struct ZoteroPDF: Identifiable, Sendable, Hashable {
    let attachmentKey: String
    let parentKey: String?
    let title: String
    let authorNames: [String]
    let year: String?
    let doi: String?
    let fileURL: URL
    let dateAdded: String
    var fingerprintPrefix: String?
    var isIndexed: Bool

    var id: String { attachmentKey }
    var authors: String { authorNames.joined(separator: ", ") }

    var importMetadata: ImportMetadata {
        ImportMetadata(
            title: title,
            authors: authorNames,
            year: year.flatMap(Int.init),
            doi: doi?.nonBlank,
            zoteroItemKey: parentKey,
            zoteroAttachmentKey: attachmentKey
        )
    }
}

extension String {
    var nonBlank: String? {
        trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : self
    }
}
