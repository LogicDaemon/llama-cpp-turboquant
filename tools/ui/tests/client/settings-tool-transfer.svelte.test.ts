import {
	ALWAYS_ALLOWED_TOOLS_LOCALSTORAGE_KEY,
	DISABLED_TOOL_CATEGORIES_LOCALSTORAGE_KEY,
	DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY
} from '$lib/constants';
import { ToolSource } from '$lib/enums';
import { exportSettings, importSettings } from '$lib/services/settings-transfer.service';
import { permissionsStore } from '$lib/stores/permissions.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import { toolsStore } from '$lib/stores/tools.svelte';
import { beforeEach, describe, expect, it } from 'vitest';

describe('settings tool transfer', () => {
	beforeEach(() => {
		toolsStore.setDisabledTools([]);
		toolsStore.setDisabledToolCategories([]);
		permissionsStore.setTools([]);
	});

	it('round-trips tool choices and persists them', () => {
		toolsStore.setDisabledTools(['mcp-server:search', toolsStore.getMcpServerToolsKey('server')]);
		toolsStore.setDisabledToolCategories([ToolSource.BROWSER]);
		permissionsStore.setTools(['mcp-server:read', 'server:read_file']);
		const exported = exportSettings();

		toolsStore.setDisabledTools(['custom:other']);
		toolsStore.setDisabledToolCategories([]);
		permissionsStore.setTools(['custom:other']);
		importSettings(JSON.parse(JSON.stringify(exported)));

		expect([...toolsStore.disabledTools]).toEqual(exported.disabledTools);
		expect([...toolsStore.disabledToolCategories]).toEqual(exported.disabledToolCategories);
		expect([...permissionsStore.tools]).toEqual(exported.alwaysAllowedTools);
		expect(JSON.parse(localStorage.getItem(DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY)!)).toEqual(
			exported.disabledTools
		);
		expect(JSON.parse(localStorage.getItem(DISABLED_TOOL_CATEGORIES_LOCALSTORAGE_KEY)!)).toEqual(
			exported.disabledToolCategories
		);
		expect(JSON.parse(localStorage.getItem(ALWAYS_ALLOWED_TOOLS_LOCALSTORAGE_KEY)!)).toEqual(
			exported.alwaysAllowedTools
		);
	});

	it('clears existing tool choices when the export contains empty lists', () => {
		const exported = exportSettings();

		toolsStore.setDisabledTools(['mcp-server:search']);
		toolsStore.setDisabledToolCategories([ToolSource.BROWSER]);
		permissionsStore.setTools(['mcp-server:read']);
		importSettings(exported);

		expect([...toolsStore.disabledTools]).toEqual([]);
		expect([...toolsStore.disabledToolCategories]).toEqual([]);
		expect([...permissionsStore.tools]).toEqual([]);
	});

	it('preserves tool choices when importing an older export', () => {
		toolsStore.setDisabledTools(['mcp-server:search']);
		permissionsStore.setTools(['mcp-server:read']);
		importSettings(settingsStore.exportSettings());
		expect([...toolsStore.disabledTools]).toEqual(['mcp-server:search']);
		expect([...permissionsStore.tools]).toEqual(['mcp-server:read']);
	});

	it('rejects invalid tool settings before changing existing settings', () => {
		toolsStore.setDisabledTools(['mcp-server:search']);
		const data = exportSettings();

		data.alwaysAllowedTools = ['mcp-server:read', 1] as unknown as string[];
		expect(() => importSettings(data)).toThrow();
		expect([...toolsStore.disabledTools]).toEqual(['mcp-server:search']);
	});
});
