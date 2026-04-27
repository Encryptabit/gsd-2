export function extractSubagentAgentClasses(input: unknown): string[] {
  if (!input || typeof input !== "object") return [];

  const record = input as Record<string, unknown>;
  const agentClasses: string[] = [];
  const addAgentClass = (value: unknown): void => {
    if (typeof value === "string" && value.trim().length > 0) agentClasses.push(value.trim());
  };
  const addFromItems = (value: unknown): void => {
    if (!Array.isArray(value)) return;
    for (const item of value) {
      if (item && typeof item === "object") addAgentClass((item as Record<string, unknown>).agent);
    }
  };

  addAgentClass(record.agent);
  addFromItems(record.tasks);
  addFromItems(record.chain);
  return agentClasses;
}
