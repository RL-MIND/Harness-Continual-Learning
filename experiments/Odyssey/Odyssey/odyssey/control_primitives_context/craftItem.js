// count is recipe executions, not desired output quantity.
// Craft 8 oak_planks from 2 oak_log: craftItem(bot, "oak_planks", 2);
// You must place a crafting table before calling this function
async function craftItem(bot, name, count = 1) {
    const item = mcData.itemsByName[name];
    const craftingTable = bot.findBlock({
        matching: mcData.blocksByName.crafting_table.id,
        maxDistance: 32,
    });
    await bot.pathfinder.goto(
        new GoalLookAtBlock(craftingTable.position, bot.world)
    );
    const available = new Map();
    for (const inventoryItem of bot.inventory.items()) {
        available.set(inventoryItem.type, (available.get(inventoryItem.type) || 0) + inventoryItem.count);
    }
    const recipes = bot.recipesFor(item.id, null, 1, craftingTable);
    const recipe = recipes.find((candidate) => {
        const required = new Map();
        for (const delta of candidate.delta || []) {
            if (delta.count < 0) required.set(delta.id, (required.get(delta.id) || 0) + (-delta.count * count));
        }
        return [...required.entries()].every(([id, needed]) => (available.get(id) || 0) >= needed);
    });
    if (!recipe) throw new Error(`No craftable recipe for ${name} with current inventory`);
    await bot.craft(recipe, count, craftingTable);
}
