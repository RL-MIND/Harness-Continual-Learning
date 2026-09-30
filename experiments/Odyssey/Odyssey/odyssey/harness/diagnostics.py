from __future__ import annotations


DIAGNOSTIC_SKILLS = {
    "diagnoseInventoryAndNearby": {
        "description": (
            "Name: diagnoseInventoryAndNearby; Description: Inspect inventory and nearby common blocks "
            "without intentionally changing the world. Use when the router is uncertain about resources, "
            "placed crafting tables, wood, stone, or basic task preconditions.\n"
        ),
        "code": """async function diagnoseInventoryAndNearby(bot) {
  const interestingBlocks = ["crafting_table", "oak_log", "birch_log", "spruce_log", "jungle_log", "acacia_log", "dark_oak_log", "mangrove_log", "stone", "chest", "furnace"];
  const nearby = [];
  for (const name of interestingBlocks) {
    const block = mcData.blocksByName[name] ? bot.findBlock({
      matching: mcData.blocksByName[name].id,
      maxDistance: 32
    }) : null;
    if (block) {
      nearby.push(`${name}@${block.position}`);
    }
  }
  const inventory = bot.inventory.items().map(item => `${item.name}:${item.count}`).join(", ") || "empty";
  bot.chat(`diagnostic inventory=${inventory}`);
  bot.chat(`diagnostic nearby=${nearby.join(", ") || "none"}`);
}""",
        "metadata": {"builtin": True, "type": "diagnostic"},
    }
}
