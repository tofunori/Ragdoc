import Foundation

/// Persistent acknowledgements are distinct from notifications awaiting a decision.
struct ZoteroWatchLedger: Codable, Sendable {
    var hasBaseline = false
    var seenKeys: Set<String> = []
    var dismissedFingerprints: Set<String> = []
    var pending: [ZoteroPDF] = []

    mutating func establishBaseline(_ documents: [ZoteroPDF]) {
        hasBaseline = true
        seenKeys.formUnion(documents.map(\.attachmentKey))
    }

    func candidates(in documents: [ZoteroPDF]) -> [ZoteroPDF] {
        let pendingKeys = Set(pending.map(\.attachmentKey))
        return documents.filter { !seenKeys.contains($0.attachmentKey) || pendingKeys.contains($0.attachmentKey) }
    }

    /// Called only after successful local reads, fingerprinting and authoritative library read.
    mutating func reconcile(_ candidates: [ZoteroPDF], history: [HistoryDocument], jobs: [ImportJob], queueHashes: Set<String>) {
        let availableKeys = Set(candidates.map(\.attachmentKey))
        let unavailable = pending.filter { !availableKeys.contains($0.attachmentKey) }
        pending = unavailable + candidates.filter { document in
            guard let hash = document.fingerprint, !dismissedFingerprints.contains(hash),
                  !queueHashes.contains(hash),
                  !ZoteroDeduplication.isIndexed(document, history: history),
                  !ZoteroDeduplication.isQueued(document, jobs: jobs) else { return false }
            return true
        }
        seenKeys.formUnion(candidates.map(\.attachmentKey))
    }

    /// Keep every pending alias, but expose one locally available copy per content hash.
    func availableNotifications(in documents: [ZoteroPDF]) -> [ZoteroPDF] {
        let available = Set(documents.map(\.attachmentKey))
        var included: Set<String> = []
        return pending.filter { document in
            guard available.contains(document.attachmentKey), let hash = document.fingerprint else { return false }
            return included.insert(hash).inserted
        }
    }

    mutating func dismiss(_ keys: Set<String>) {
        dismissedFingerprints.formUnion(pending.filter { keys.contains($0.attachmentKey) }.compactMap(\.fingerprint))
        seenKeys.formUnion(keys)
        pending.removeAll { keys.contains($0.attachmentKey) || $0.fingerprint.map { dismissedFingerprints.contains($0) } == true }
    }
}

enum ZoteroDeduplication {
    static func isIndexed(_ document: ZoteroPDF, history: [HistoryDocument]) -> Bool {
        history.contains { item in
            item.zoteroAttachmentKey == document.attachmentKey
                || (document.fingerprint != nil && item.pdfFingerprint == document.fingerprint)
                || item.source.hasSuffix("_\(document.attachmentKey).md")
                || document.fingerprintPrefix.map { prefix in
                    !prefix.isEmpty && item.source.hasSuffix("_\(prefix).md")
                } == true
        }
    }

    static func isQueued(_ document: ZoteroPDF, jobs: [ImportJob]) -> Bool {
        jobs.contains {
            $0.fileURL.standardizedFileURL == document.fileURL.standardizedFileURL
                || $0.metadata?.zoteroAttachmentKey == document.attachmentKey
                || (document.fingerprint != nil && $0.fingerprint == document.fingerprint)
        }
    }
}
