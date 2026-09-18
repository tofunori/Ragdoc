import Foundation

enum HistoryServiceError: LocalizedError {
    case commandFailed(String)
    case invalidResponse

    var errorDescription: String? {
        switch self {
        case .commandFailed(let details): "Reading Ragdoc failed. \(details)"
        case .invalidResponse: "Ragdoc returned unreadable library data."
        }
    }
}

struct RagdocHistoryService: Sendable {
    let configuration: PipelineConfiguration

    func fetchDocuments(includePDFIdentity: Bool = false) async throws -> [HistoryDocument] {
        try configuration.validate()
        if configuration.location == .local {
            return try await LocalEngine.current.decoded("documents", connection: configuration.localConnection)
        }
        return try await Task.detached(priority: .userInitiated) {
            let root = configuration.remoteRoot
            let python = """
            import json,sqlite3
            path='\(root)/chroma_db_new/chroma.sqlite3'
            con=sqlite3.connect(f'file:{path}?mode=ro',uri=True,timeout=5)
            con.execute('PRAGMA query_only=ON')
            con.execute('BEGIN')
            collection=con.execute('SELECT id FROM collections WHERE name=?',('ragdoc_contextualized_v1',)).fetchone()
            if not collection: raise RuntimeError('Ragdoc collection not found')
            collection_id=collection[0]
            state={r[0]:(r[1] if r[1] is not None else r[2] if r[2] is not None else r[3] if r[3] is not None else bool(r[4])) for r in con.execute("SELECT key,str_value,int_value,float_value,bool_value FROM collection_metadata WHERE collection_id=? AND key IN ('ragdoc_revision','ragdoc_write_state','ragdoc_repairing')",(collection_id,))}
            if state.get('ragdoc_write_state','ready')!='ready' or state.get('ragdoc_repairing',False): raise RuntimeError('Ragdoc indexing in progress; try again shortly.')
            query='''
            WITH item AS (
             SELECT em.id,
                    MAX(CASE WHEN em.key='source' THEN em.string_value END) AS source,
                    MAX(CASE WHEN em.key='title' THEN em.string_value END) AS title,
                    MAX(CASE WHEN em.key='indexed_date' THEN em.string_value END) AS indexed_date,
                    MAX(CASE WHEN em.key='doi' THEN em.string_value END) AS doi
             FROM embedding_metadata em
             JOIN embeddings e ON e.id=em.id
             JOIN segments s ON s.id=e.segment_id
             WHERE s.collection=? AND em.key IN ('source','title','indexed_date','doi')
             GROUP BY em.id
            )
            SELECT source,COALESCE(MAX(NULLIF(title,'')),source),COUNT(*),MAX(indexed_date),MAX(NULLIF(doi,''))
            FROM item WHERE source IS NOT NULL
            GROUP BY source
            ORDER BY COALESCE(MAX(indexed_date),'') DESC,source DESC
            '''
            rows=[{'source':r[0],'title':r[1],'chunks':r[2],'indexedDate':r[3],'doi':r[4]} for r in con.execute(query,(collection_id,))]
            if \(includePDFIdentity ? "True" : "False"):
                from pathlib import Path
                for row in rows:
                    source=Path(row['source'])
                    if source.name!=row['source']: continue
                    sidecar=Path('\(root)')/'articles_markdown'/source.with_suffix('.metadata.json')
                    try:
                        identity=json.loads(sidecar.read_text())
                        row['pdfFingerprint']=identity.get('pdf_sha256')
                        row['zoteroAttachmentKey']=identity.get('zotero_attachment_key')
                    except (OSError,ValueError): pass
            con.rollback()
            con.close()
            print(json.dumps(rows,ensure_ascii=False))
            """
            let command = "cd \(shellQuote(root)) && /usr/bin/timeout 20s ./ragdoc-env-new/bin/python3 -c \(shellQuote(python))"

            let temporary = FileManager.default.temporaryDirectory
            let outputURL = temporary.appendingPathComponent("ragdrop-history-\(UUID().uuidString).json")
            let errorURL = temporary.appendingPathComponent("ragdrop-history-\(UUID().uuidString).err")
            FileManager.default.createFile(atPath: outputURL.path, contents: nil)
            FileManager.default.createFile(atPath: errorURL.path, contents: nil)
            defer {
                try? FileManager.default.removeItem(at: outputURL)
                try? FileManager.default.removeItem(at: errorURL)
            }

            let outputHandle = try FileHandle(forWritingTo: outputURL)
            let errorHandle = try FileHandle(forWritingTo: errorURL)
            defer {
                try? outputHandle.close()
                try? errorHandle.close()
            }

            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/ssh")
            process.arguments = [
                "-o", "ControlMaster=no",
                "-o", "ControlPath=none",
                "-o", "ConnectTimeout=8",
                "-o", "ServerAliveInterval=5",
                "-o", "ServerAliveCountMax=2",
                configuration.nasHost,
                command
            ]
            process.standardOutput = outputHandle
            process.standardError = errorHandle
            try process.run()
            process.waitUntilExit()
            try? outputHandle.synchronize()
            try? errorHandle.synchronize()

            guard process.terminationStatus == 0 else {
                let details = String(decoding: (try? Data(contentsOf: errorURL)) ?? Data(), as: UTF8.self)
                throw HistoryServiceError.commandFailed(String(details.suffix(900)))
            }
            guard let documents = try? JSONDecoder().decode(
                [HistoryDocument].self,
                from: Data(contentsOf: outputURL)
            ) else {
                throw HistoryServiceError.invalidResponse
            }
            return documents
        }.value
    }

    private func shellQuote(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
}
