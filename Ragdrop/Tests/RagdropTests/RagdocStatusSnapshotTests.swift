import Foundation
import Testing
@testable import Ragdrop

@Suite("Ragdoc status")
struct RagdocStatusSnapshotTests {
    @Test func acceptsACompleteHealthySnapshot() throws {
        let data = #"{"tools":["semantic_search_hybrid","search_by_source","list_documents","get_document_content","get_indexation_status","get_server_status"],"searchOK":true,"mcpError":null,"latencySeconds":2.4,"documents":500,"chunks":70504,"revision":"abc","writeState":"ready","repairing":false,"models":{"voyage-context-4":500},"expectedModel":"voyage-context-4","serverModel":"voyage-context-4","serverRevision":"abc","lexicalReady":true,"rerankingModel":"rerank-v4.0-pro","listenerCount":1,"testSource":"paper.md"}"#.data(using: .utf8)!
        let snapshot = try JSONDecoder().decode(RagdocStatusSnapshot.self, from: data)
        #expect(snapshot.isHealthy)
        #expect(snapshot.missingTools.isEmpty)
    }

    @Test func rejectsARespondingServerDuringRepair() throws {
        let snapshot = RagdocStatusSnapshot(
            tools: ["semantic_search_hybrid", "search_by_source", "list_documents", "get_document_content", "get_indexation_status", "get_server_status"],
            searchOK: true,
            mcpError: nil,
            latencySeconds: 1,
            documents: 500,
            chunks: 70504,
            revision: "abc",
            writeState: "writing",
            repairing: true,
            models: ["voyage-context-4": 500],
            expectedModel: "voyage-context-4",
            serverModel: "voyage-context-4",
            serverRevision: "abc",
            lexicalReady: true,
            rerankingModel: "rerank-v4.0-pro",
            listenerCount: 1,
            testSource: "paper.md"
        )
        #expect(!snapshot.isHealthy)
    }

    @Test func rejectsAnUnexpectedEmbeddingModel() {
        let snapshot = RagdocStatusSnapshot(
            tools: ["semantic_search_hybrid", "search_by_source", "list_documents", "get_document_content", "get_indexation_status", "get_server_status"],
            searchOK: true,
            mcpError: nil,
            latencySeconds: 1,
            documents: 500,
            chunks: 70504,
            revision: "abc",
            writeState: "ready",
            repairing: false,
            models: ["unknown": 500],
            expectedModel: "voyage-context-4",
            serverModel: "voyage-context-4",
            serverRevision: "abc",
            lexicalReady: true,
            rerankingModel: "rerank-v4.0-pro",
            listenerCount: 1,
            testSource: "paper.md"
        )
        #expect(!snapshot.isHealthy)
    }

    @Test func rejectsAMissingLexicalIndex() {
        let snapshot = RagdocStatusSnapshot(
            tools: ["semantic_search_hybrid", "search_by_source", "list_documents", "get_document_content", "get_indexation_status", "get_server_status"],
            searchOK: true,
            mcpError: nil,
            latencySeconds: 1,
            documents: 500,
            chunks: 70504,
            revision: "abc",
            writeState: "ready",
            repairing: false,
            models: ["voyage-context-4": 500],
            expectedModel: "voyage-context-4",
            serverModel: "voyage-context-4",
            serverRevision: "abc",
            lexicalReady: false,
            rerankingModel: "rerank-v4.0-pro",
            listenerCount: 1,
            testSource: "paper.md"
        )
        #expect(!snapshot.isHealthy)
    }
}
