import SwiftUI

struct QueueRowView: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    let job: ImportJob
    var onPreview: (() -> Void)?
    var onRetry: (() -> Void)?
    var onShowError: (() -> Void)?
    var onRemove: (() -> Void)?
    var canRemove = true

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: hasRecoverableError ? "exclamationmark.triangle.fill" : job.stage.symbol)
                .font(.title3)
                .foregroundStyle(stageColor)
                .frame(width: 24)
                .symbolEffect(.pulse, isActive: job.stage.isActive && !reduceMotion)

            VStack(alignment: .leading, spacing: 3) {
                Text(job.metadata?.title.nonBlank ?? job.displayName).font(.body.weight(.medium))
                    .lineLimit(1)
                Text(job.detail)
                    .font(.caption)
                    .foregroundStyle(hasRecoverableError || job.stage == .failed ? RagdropTheme.error : RagdropTheme.secondary)
                    .lineLimit(2)
            }
            Spacer(minLength: 12)
            if job.stage == .awaitingReview || job.stage == .readyForIndexing {
                Button("Réviser", systemImage: "rectangle.split.2x1", action: { onPreview?() })
                    .help("Comparer le PDF original et le texte extrait")
            }
            if job.errorDetails != nil {
                Button("Détails", systemImage: "info.circle", action: { onShowError?() })
                    .labelStyle(.iconOnly)
                    .help("Afficher le détail de l’erreur")
            }
            if job.stage == .failed {
                Button("Relancer", systemImage: "arrow.clockwise", action: { onRetry?() })
                    .labelStyle(.iconOnly)
                    .help("Relancer ce PDF")
            }
            Button("Retirer", systemImage: "trash", role: .destructive, action: { onRemove?() })
                .labelStyle(.iconOnly)
                .help("Retirer de la file sans supprimer l’article de Ragdoc")
                .disabled(!canRemove)
            Text(hasRecoverableError ? "À reprendre" : job.stage.title)
                .font(.caption.weight(.medium))
                .foregroundStyle(stageColor)
        }
        .padding(.vertical, 5)
    }

    private var stageColor: Color {
        if hasRecoverableError {
            return RagdropTheme.error
        }
        switch job.stage {
        case .completed: return RagdropTheme.success
        case .awaitingReview: return RagdropTheme.warning
        case .readyForIndexing: return RagdropTheme.success
        case .duplicate: return RagdropTheme.warning
        case .rejected: return RagdropTheme.secondary
        case .failed: return RagdropTheme.error
        case .queued: return RagdropTheme.secondary
        default: return RagdropTheme.accent
        }
    }

    private var hasRecoverableError: Bool {
        job.errorDetails != nil && job.stage == .readyForIndexing
    }
}
