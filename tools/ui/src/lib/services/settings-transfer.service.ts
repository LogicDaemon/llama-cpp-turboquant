import { ToolSource } from '$lib/enums';
import { permissionsStore } from '$lib/stores/permissions.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import { toolsStore } from '$lib/stores/tools.svelte';
import type { SettingsExportType } from '$lib/types';

export function exportSettings(includeSensitiveData: boolean = false): SettingsExportType {
	return {
		...settingsStore.exportSettings(includeSensitiveData),
		alwaysAllowedTools: [...permissionsStore.tools],
		disabledToolCategories: [...toolsStore.disabledToolCategories],
		disabledTools: [...toolsStore.disabledTools]
	};
}

export function importSettings(data: SettingsExportType): void {
	for (const value of [data.disabledTools, data.disabledToolCategories, data.alwaysAllowedTools]) {
		if (
			value !== undefined &&
			(!Array.isArray(value) || !value.every((key) => typeof key === 'string'))
		) {
			throw new Error('Invalid settings data: tool settings must be arrays of strings');
		}
	}

	if (
		data.disabledToolCategories?.some(
			(source) => !Object.values(ToolSource).includes(source as ToolSource)
		)
	) {
		throw new Error('Invalid settings data: unknown tool category');
	}

	settingsStore.importSettings(data);

	if (data.disabledTools !== undefined) toolsStore.setDisabledTools(data.disabledTools);

	if (data.disabledToolCategories !== undefined) {
		toolsStore.setDisabledToolCategories(data.disabledToolCategories as ToolSource[]);
	}

	if (data.alwaysAllowedTools !== undefined) permissionsStore.setTools(data.alwaysAllowedTools);
}
