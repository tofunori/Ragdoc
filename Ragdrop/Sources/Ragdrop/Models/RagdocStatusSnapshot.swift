import Foundation

struct RagdocStatusSnapshot: Codable, Sendable {
    let tools: [String]
    let searchOK: Bool
    let mcpError: String?
    let latencySeconds: Double?
    let documents: Int
    let chunks: Int
    let revision: String?
    let writeState: String
    let repairing: Bool
    let models: [String: Int]
    let expectedModel: String
    let serverModel: String?
    let serverRevision: String?
    let lexicalReady: Bool
    let rerankingModel: String?
    let listenerCount: Int
    let testSource: String?

    private static let requiredTools: Set<String> = [
        "semantic_search_hybrid",
        "search_by_source",
        "list_documents",
        "get_document_content",
        "get_indexation_status",
        "get_server_status"
    ]

    var missingTools: [String] {
        Self.requiredTools.subtracting(tools).sorted()
    }

    var isHealthy: Bool {
        mcpError == nil
            && searchOK
            && writeState == "ready"
            && !repairing
            && listenerCount == 1
            && missingTools.isEmpty
            && documents > 0
            && chunks > 0
            && models.count == 1
            && models[expectedModel] == documents
            && serverModel == expectedModel
            && serverRevision == revision
            && lexicalReady
            && rerankingModel == "rerank-v4.0-pro"
    }

    var embeddingSummary: String {
        models
            .sorted { $0.key < $1.key }
            .map { "\($0.key) · \($0.value) documents" }
            .joined(separator: "\n")
    }
}
