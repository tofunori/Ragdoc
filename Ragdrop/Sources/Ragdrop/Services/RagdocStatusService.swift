import Foundation

enum RagdocStatusError: LocalizedError {
    case commandFailed(String)
    case invalidResponse

    var errorDescription: String? {
        switch self {
        case .commandFailed(let details): "The Ragdoc diagnostic failed. \(details)"
        case .invalidResponse: "The Ragdoc diagnostic returned an unreadable response."
        }
    }
}

struct RagdocStatusService: Sendable {
    let configuration: PipelineConfiguration

    func check() async throws -> RagdocStatusSnapshot {
        try configuration.validate()
        return try await Task.detached(priority: .userInitiated) {
            let root = configuration.remoteRoot
            let python = """
            import asyncio,json,sqlite3,subprocess,time
            from fastmcp import Client
            path='\(root)/chroma_db_new/chroma.sqlite3'
            con=sqlite3.connect(f'file:{path}?mode=ro',uri=True,timeout=5)
            con.execute('PRAGMA query_only=ON')
            con.execute('BEGIN')
            collection=con.execute('SELECT id FROM collections WHERE name=?',('ragdoc_contextualized_v1',)).fetchone()
            if not collection: raise RuntimeError('Ragdoc collection not found')
            collection_id=collection[0]
            def value(row):
                return row[1] if row[1] is not None else row[2] if row[2] is not None else row[3] if row[3] is not None else bool(row[4])
            metadata={r[0]:value(r) for r in con.execute('SELECT key,str_value,int_value,float_value,bool_value FROM collection_metadata WHERE collection_id=?',(collection_id,))}
            segment=con.execute("SELECT id FROM segments WHERE collection=? AND scope='METADATA'",(collection_id,)).fetchone()[0]
            documents=con.execute("SELECT COUNT(DISTINCT string_value) FROM embedding_metadata WHERE key='source' AND id IN (SELECT id FROM embeddings WHERE segment_id=?)",(segment,)).fetchone()[0]
            chunks=con.execute('SELECT COUNT(*) FROM embeddings WHERE segment_id=?',(segment,)).fetchone()[0]
            models=dict(con.execute("SELECT model.string_value,COUNT(DISTINCT source.string_value) FROM embedding_metadata model JOIN embedding_metadata source ON source.id=model.id AND source.key='source' JOIN embeddings embedding ON embedding.id=model.id WHERE model.key='model' AND embedding.segment_id=? GROUP BY model.string_value",(segment,)))
            latest=con.execute("SELECT source.string_value,MAX(date.string_value) FROM embedding_metadata source JOIN embedding_metadata date ON date.id=source.id AND date.key='indexed_date' JOIN embeddings embedding ON embedding.id=source.id WHERE source.key='source' AND embedding.segment_id=? GROUP BY source.string_value ORDER BY MAX(date.string_value) DESC LIMIT 1",(segment,)).fetchone()
            latest=latest[0] if latest else None
            con.rollback()
            con.close()
            async def mcp_check():
                started=time.monotonic()
                async with Client('http://127.0.0.1:8484/mcp') as client:
                    tools=[tool.name for tool in await client.list_tools()]
                    status_result=await client.call_tool('get_server_status',{})
                    server_status=getattr(status_result,'data',{}) or {}
                    if not latest: return tools,False,round(time.monotonic()-started,2),server_status
                    result=await client.call_tool('search_by_source',{'query':'main finding','sources':[latest],'top_k':1,'alpha':0.5,'multi_query':False,'format':'compact','preview_chars':80})
                    text=getattr(result,'data','') or ''
                    has_hit='[1]' in text and f'source={latest}' in text
                    hybrid_pipeline='retrieval: hybrid' in text and 'bm25_rank=None semantic_rank=None' not in text
                    reranking_pipeline='Reranking: cohere' in text and 'rerank=unavailable' not in text
                    success=not bool(getattr(result,'is_error',False)) and has_hit and hybrid_pipeline and reranking_pipeline and '[!]' not in text
                    return tools,success,round(time.monotonic()-started,2),server_status
            tools=[]
            search_ok=False
            latency=None
            mcp_error=None
            server_status={}
            try:
                tools,search_ok,latency,server_status=asyncio.run(mcp_check())
            except Exception as error:
                mcp_error=str(error)[:500]
            sockets=subprocess.run(['ss','-lnt'],capture_output=True,text=True,timeout=5).stdout
            listeners=[line for line in sockets.splitlines() if 'LISTEN' in line and ':8484' in line]
            expected_model=metadata.get('embedding_model') or (next(iter(models)) if len(models)==1 else 'inconnu')
            lexical=server_status.get('lexical_index') or {}
            print(json.dumps({'tools':tools,'searchOK':search_ok,'mcpError':mcp_error,'latencySeconds':latency,'documents':documents,'chunks':chunks,'revision':metadata.get('ragdoc_revision'),'writeState':metadata.get('ragdoc_write_state','legacy'),'repairing':bool(metadata.get('ragdoc_repairing',False)),'models':models,'expectedModel':expected_model,'serverModel':server_status.get('configured_embedding_model'),'serverRevision':server_status.get('index_revision'),'lexicalReady':bool(lexical.get('ready')),'rerankingModel':server_status.get('reranking_model'),'listenerCount':len(listeners),'testSource':latest},ensure_ascii=False))
            """
            let command = "cd \(shellQuote(root)) && /usr/bin/timeout 60s ./ragdoc-env-new/bin/python3 -c \(shellQuote(python))"
            let temporary = FileManager.default.temporaryDirectory
            let outputURL = temporary.appendingPathComponent("ragdrop-status-\(UUID().uuidString).json")
            let errorURL = temporary.appendingPathComponent("ragdrop-status-\(UUID().uuidString).err")
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
                throw RagdocStatusError.commandFailed(String(details.suffix(900)))
            }
            guard let snapshot = try? JSONDecoder().decode(
                RagdocStatusSnapshot.self,
                from: Data(contentsOf: outputURL)
            ) else {
                throw RagdocStatusError.invalidResponse
            }
            return snapshot
        }.value
    }

    private func shellQuote(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
}
