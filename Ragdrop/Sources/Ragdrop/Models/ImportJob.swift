import Foundation

struct ImportMetadata: Codable, Equatable, Sendable {
    let title: String
    let authors: [String]
    let year: Int?
    let doi: String?
    let zoteroItemKey: String?
    let zoteroAttachmentKey: String?
}

enum ImportStage: String, Codable, Sendable {
    case queued
    case checkingDuplicate
    case converting
    case awaitingReview
    case readyForIndexing
    case transferring
    case indexing
    case verifying
    case completed
    case duplicate
    case rejected
    case failed

    var title: String {
        switch self {
        case .queued: "Queued"
        case .checkingDuplicate: "Checking for duplicates"
        case .converting: "PDF conversion"
        case .awaitingReview: "Review needed"
        case .readyForIndexing: "Approved"
        case .transferring: "Transfer to server"
        case .indexing: "Ragdoc indexing"
        case .verifying: "Passage verification"
        case .completed: "Added"
        case .duplicate: "Already present"
        case .rejected: "Rejected"
        case .failed: "Failed"
        }
    }

    var symbol: String {
        switch self {
        case .queued: "clock"
        case .checkingDuplicate: "doc.on.doc"
        case .converting: "doc.text.magnifyingglass"
        case .awaitingReview: "eye.circle"
        case .readyForIndexing: "checkmark.circle"
        case .transferring: "arrow.up.circle"
        case .indexing: "square.stack.3d.up"
        case .verifying: "checkmark.shield"
        case .completed: "checkmark.circle.fill"
        case .duplicate: "doc.on.doc.fill"
        case .rejected: "xmark.circle"
        case .failed: "exclamationmark.triangle.fill"
        }
    }

    var isActive: Bool {
        switch self {
        case .checkingDuplicate, .converting, .transferring, .indexing, .verifying: true
        default: false
        }
    }

    var ownsFingerprint: Bool {
        switch self {
        case .awaitingReview, .readyForIndexing, .transferring, .indexing, .verifying, .completed:
            true
        default:
            false
        }
    }

    var progressFraction: Double? {
        switch self {
        case .queued: 0
        case .checkingDuplicate: 0.04
        case .converting: 0.12
        case .awaitingReview: 0.30
        case .readyForIndexing: 0.35
        case .transferring: 0.40
        case .indexing: 0.66
        case .verifying: 0.88
        case .completed, .duplicate, .rejected: 1
        case .failed: nil
        }
    }
}

struct ImportJob: Codable, Identifiable, Equatable, Sendable {
    let id: UUID
    let fileURL: URL
    var stage: ImportStage
    var detail: String
    var stageStartedAt: Date
    var artifactURL: URL?
    var metadataURL: URL?
    var visualArtifactBundleURL: URL?
    var visualArtifactCount = 0
    var chunkCount: Int?
    var fingerprint: String?
    var metadata: ImportMetadata?
    var errorDetails: String?
    var progressHighWater = 0.0
    // Optional so queues saved before converter tracking remain readable.
    var converterName: String?

    init(fileURL: URL, metadata: ImportMetadata? = nil) {
        id = UUID()
        self.fileURL = fileURL.standardizedFileURL
        stage = .queued
        detail = "Ready to add"
        stageStartedAt = Date()
        self.metadata = metadata
    }

    var displayName: String {
        fileURL.deletingPathExtension().lastPathComponent
    }

    var overallProgress: Double {
        max(progressHighWater, stage.progressFraction ?? progressHighWater)
    }

    mutating func recordProgress(for stage: ImportStage) {
        if let progress = stage.progressFraction {
            progressHighWater = max(progressHighWater, progress)
        }
    }
}
