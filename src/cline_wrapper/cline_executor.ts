import * as path from 'path';
import { AutoApprovalSettings } from './auto_approval_settings';
import { requestConfirmation } from '../cli_confirmation';

export interface ActionDetails {
    type: string; // e.g., "editFile", "executeCommand", "readFile", "listDirectory", "useBrowser", "useMcp"
    filePath?: string;
    command?: string;
    url?: string;
    toolName?: string; // For MCP
    serverName?: string; // For MCP
    isSafeCommand?: boolean; // For executeCommand
}

function isExternalPath(filePath: string | undefined): boolean {
    if (!filePath) {
        return false; // Or handle as an error, depending on requirements
    }
    const absolutePath = path.resolve(filePath);
    const cwd = process.cwd();
    return !absolutePath.startsWith(cwd);
}

export async function executeClineAction(
    actionDetails: ActionDetails,
    approvalSettings: AutoApprovalSettings,
): Promise<any> {
    const { actions } = approvalSettings;
    let needsConfirmation = false;
    let confirmationPrompt = '';

    switch (actionDetails.type) {
        case 'readFile':
        case 'listDirectory': // Assuming listDirectory falls under readFile permissions
            if (isExternalPath(actionDetails.filePath)) {
                if (!actions.cliAllowReadFilesExternally) {
                    needsConfirmation = true;
                    confirmationPrompt = `Allow reading ${actionDetails.filePath}?`;
                } else if (!actions.cliAllowReadFiles) { // Still need base read permission
                     needsConfirmation = true;
                     confirmationPrompt = `Allow reading ${actionDetails.filePath}? (external allowed, base read needed)`;
                }
            } else {
                if (!actions.cliAllowReadFiles) {
                    needsConfirmation = true;
                    confirmationPrompt = `Allow reading ${actionDetails.filePath || 'directory'}?`;
                }
            }
            break;

        case 'editFile':
        case 'newFile': // Assuming newFile falls under editFile permissions
            if (isExternalPath(actionDetails.filePath)) {
                if (!actions.cliAllowEditFilesExternally) {
                    needsConfirmation = true;
                    confirmationPrompt = `Allow editing ${actionDetails.filePath}?`;
                } else if (!actions.cliAllowEditFiles) { // Still need base edit permission
                    needsConfirmation = true;
                    confirmationPrompt = `Allow editing ${actionDetails.filePath}? (external allowed, base edit needed)`;
                }
            } else {
                if (!actions.cliAllowEditFiles) {
                    needsConfirmation = true;
                    confirmationPrompt = `Allow editing ${actionDetails.filePath}?`;
                }
            }
            break;

        case 'executeCommand':
            if (actionDetails.isSafeCommand) {
                if (!actions.cliAllowExecuteSafeCommands) {
                    needsConfirmation = true;
                    confirmationPrompt = `Allow executing safe command: ${actionDetails.command}?`;
                }
            } else {
                if (!actions.cliAllowExecuteAllCommands) {
                    needsConfirmation = true;
                    confirmationPrompt = `Allow executing command: ${actionDetails.command}?`;
                }
            }
            break;

        case 'useBrowser':
            if (!actions.cliAllowUseBrowser) {
                needsConfirmation = true;
                confirmationPrompt = `Allow using browser for URL: ${actionDetails.url}?`;
            }
            break;

        case 'useMcp':
            if (!actions.cliAllowUseMcp) {
                needsConfirmation = true;
                confirmationPrompt = `Allow using MCP tool: ${actionDetails.toolName} on server ${actionDetails.serverName}?`;
            }
            break;

        default:
            return { status: 'error', message: `Unknown action type: ${actionDetails.type}` };
    }

    if (needsConfirmation) {
        const confirmed = await requestConfirmation(confirmationPrompt);
        if (!confirmed) {
            return { status: 'denied', message: `User denied action: ${actionDetails.type}` };
        }
    }

    // If execution reaches here, the action is approved (either by flag or by prompt)
    // TODO: Invoke actual Cline tool for ${actionDetails.type}
    console.log(`Action approved: ${actionDetails.type} for ${actionDetails.filePath || actionDetails.command || actionDetails.url}`);
    return { status: 'approved', message: `Action would proceed for ${actionDetails.type}.` };
}
