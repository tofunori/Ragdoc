import CryptoKit
import Foundation

enum ZoteroLibraryError: LocalizedError {
    case unavailable
    case invalidResponse

    var errorDescription: String? {
        switch self {
        case .unavailable: "Ouvrez Zotero pour lire votre bibliothèque locale."
        case .invalidResponse: "Zotero a renvoyé une bibliothèque illisible."
        }
    }
}

struct ZoteroLibraryService: Sendable {
    private let itemsEndpoint = URL(string: "http://127.0.0.1:23119/api/users/0/items")!
    private let topItemsEndpoint = URL(string: "http://127.0.0.1:23119/api/users/0/items/top")!

    func fetchPDFs(includeFingerprints: Bool = true, onlyKeys: Set<String>? = nil) async throws -> [ZoteroPDF] {
        var attachments: [ZoteroAPIItem] = []
        for start in stride(from: 0, to: 100_000, by: 100) {
            let page = try await fetch(from: itemsEndpoint, query: [
                URLQueryItem(name: "itemType", value: "attachment"),
                URLQueryItem(name: "limit", value: "100"),
                URLQueryItem(name: "start", value: String(start))
            ])
            attachments.append(contentsOf: page)
            if page.count < 100 { break }
        }

        var parents: [String: ZoteroAPIItem] = [:]
        for start in stride(from: 0, to: 100_000, by: 100) {
            let page = try await fetch(from: topItemsEndpoint, query: [
                URLQueryItem(name: "limit", value: "100"),
                URLQueryItem(name: "start", value: String(start))
            ])
            for item in page { parents[item.key] = item }
            if page.count < 100 { break }
        }

        let documents: [ZoteroPDF] = attachments.compactMap { attachment -> ZoteroPDF? in
            guard onlyKeys == nil || onlyKeys!.contains(attachment.key),
                  let enclosure = attachment.links?.enclosure,
                  enclosure.type == "application/pdf",
                  let url = URL(string: enclosure.href),
                  url.isFileURL,
                  FileManager.default.isReadableFile(atPath: url.path) else { return nil }
            let parent = attachment.data.parentItem.flatMap { parents[$0] }
            let title = parent?.data.title?.nonBlank
                ?? attachment.data.filename?.nonBlank
                ?? enclosure.title?.nonBlank
                ?? url.deletingPathExtension().lastPathComponent
            let authors = parent?.data.creators?
                .filter { $0.creatorType == "author" }
                .compactMap(\.displayName)
                ?? []
            return ZoteroPDF(
                attachmentKey: attachment.key,
                parentKey: attachment.data.parentItem,
                title: title,
                authorNames: authors,
                year: Self.year(in: parent?.data.date),
                doi: parent?.data.DOI,
                fileURL: url.standardizedFileURL,
                dateAdded: attachment.data.dateAdded ?? "",
                fingerprintPrefix: nil,
                isIndexed: false
            )
        }
        .sorted { lhs, rhs in
            if lhs.dateAdded != rhs.dateAdded { return lhs.dateAdded > rhs.dateAdded }
            return lhs.title.localizedCaseInsensitiveCompare(rhs.title) == .orderedAscending
        }
        guard includeFingerprints else { return documents }
        return await Task.detached(priority: .utility) {
            documents.map { document in
                var copy = document
                copy.fingerprint = try? Self.sha256(of: document.fileURL)
                copy.fingerprintPrefix = copy.fingerprint.map { String($0.prefix(12)) }
                return copy
            }
        }.value
    }

    private func fetch(from endpoint: URL, query: [URLQueryItem]) async throws -> [ZoteroAPIItem] {
        var components = URLComponents(url: endpoint, resolvingAgainstBaseURL: false)!
        components.queryItems = query
        guard let url = components.url else { throw ZoteroLibraryError.invalidResponse }
        var request = URLRequest(url: url)
        request.timeoutInterval = 12
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw ZoteroLibraryError.unavailable
        }
        guard let http = response as? HTTPURLResponse, http.statusCode == 200,
              let items = try? JSONDecoder().decode([ZoteroAPIItem].self, from: data) else {
            throw ZoteroLibraryError.invalidResponse
        }
        return items
    }

    private static func year(in value: String?) -> String? {
        guard let value,
              let range = value.range(of: #"\b(?:1[5-9]\d{2}|20\d{2}|2100)\b"#,
                                      options: .regularExpression) else { return nil }
        return String(value[range])
    }

    static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let data = try handle.read(upToCount: 1024 * 1024), !data.isEmpty {
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

private struct ZoteroAPIItem: Decodable, Sendable {
    let key: String
    let links: Links?
    let data: ItemData

    struct Links: Decodable, Sendable {
        let enclosure: Enclosure?
    }

    struct Enclosure: Decodable, Sendable {
        let href: String
        let type: String
        let title: String?
    }

    struct ItemData: Decodable, Sendable {
        let title: String?
        let filename: String?
        let parentItem: String?
        let dateAdded: String?
        let date: String?
        let creators: [Creator]?
        let DOI: String?
    }

    struct Creator: Decodable, Sendable {
        let creatorType: String?
        let name: String?
        let firstName: String?
        let lastName: String?

        var displayName: String? {
            if let name = name?.nonBlank { return name }
            let combined = [firstName, lastName].compactMap { $0?.nonBlank }.joined(separator: " ")
            return combined.nonBlank
        }
    }
}
