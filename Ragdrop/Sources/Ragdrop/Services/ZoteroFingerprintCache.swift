import Foundation

/// Read file contents only when size or modification time changes.
actor ZoteroFingerprintCache {
    private struct Entry { let size: Int; let modified: Date?; let hash: String }
    private var entries: [URL: Entry] = [:]
    func hash(_ url: URL) throws -> String {
        let url = url.standardizedFileURL
        let values = try url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
        let size = values.fileSize ?? 0
        if let entry = entries[url], entry.size == size, entry.modified == values.contentModificationDate { return entry.hash }
        let hash = try ZoteroLibraryService.sha256(of: url)
        entries[url] = Entry(size: size, modified: values.contentModificationDate, hash: hash)
        return hash
    }
    func enrich(_ documents: [ZoteroPDF]) throws -> [ZoteroPDF] {
        try documents.map { document in
            var copy = document
            copy.fingerprint = try hash(document.fileURL)
            copy.fingerprintPrefix = copy.fingerprint.map { String($0.prefix(12)) }
            return copy
        }
    }
    func hashes(for jobs: [ImportJob]) throws -> Set<String> {
        var result: Set<String> = []
        for job in jobs {
            if let fingerprint = job.fingerprint { result.insert(fingerprint) }
            else if FileManager.default.isReadableFile(atPath: job.fileURL.path) { result.insert(try hash(job.fileURL)) }
        }
        return result
    }
}
