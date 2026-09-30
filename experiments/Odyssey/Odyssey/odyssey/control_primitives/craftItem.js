function findCraftableRecipe(bot, recipes, executions = 1) {
    const available = new Map();
    for (const item of bot.inventory.items()) {
        available.set(item.type, (available.get(item.type) || 0) + item.count);
    }
    for (const recipe of recipes || []) {
        const required = new Map();
        for (const delta of recipe.delta || []) {
            if (delta.count >= 0) continue;
            required.set(delta.id, (required.get(delta.id) || 0) + (-delta.count * executions));
        }
        if ([...required.entries()].every(([id, needed]) => (available.get(id) || 0) >= needed)) {
            return recipe;
        }
    }
    return null;
}

function craftRecipeCapacity(bot, recipe) {
    const available = new Map();
    for (const item of bot.inventory.items()) {
        available.set(item.type, (available.get(item.type) || 0) + item.count);
    }
    let capacity = Infinity;
    for (const delta of recipe.delta || []) {
        if (delta.count >= 0) continue;
        capacity = Math.min(capacity, Math.floor((available.get(delta.id) || 0) / -delta.count));
    }
    return Number.isFinite(capacity) ? capacity : 0;
}

function craftCountError(bot, name, recipes, requested) {
    let bestRecipe = null;
    let maximum = 0;
    for (const recipe of recipes || []) {
        const capacity = craftRecipeCapacity(bot, recipe);
        if (capacity > maximum) {
            maximum = capacity;
            bestRecipe = recipe;
        }
    }
    const requirements = [];
    for (const delta of (bestRecipe && bestRecipe.delta) || []) {
        if (delta.count >= 0) continue;
        const item = mcData.items[delta.id];
        requirements.push(`${item ? item.name : delta.id}=${-delta.count * requested}`);
    }
    return new Error(
        `Cannot execute ${name} recipe ${requested} times; maximum executable count is ${maximum}`
        + `${requirements.length ? `; requested ingredients: ${requirements.join(", ")}` : ""}. `
        + "craftItem count means recipe executions, not desired output items."
    );
}

async function craftItem(bot, name, count = 1) {
    // return if name is not string
    if (typeof name !== "string") {
        throw new Error("name for craftItem must be a string");
    }
    // return if count is not number
    if (typeof count !== "number" || !Number.isInteger(count) || count < 1) {
        throw new Error("count for craftItem must be a positive integer number of recipe executions");
    }
    const itemByName = mcData.itemsByName[name];
    if (!itemByName) {
        throw new Error(`No item named ${name}`);
    }
    const craftingTable = bot.findBlock({
        matching: mcData.blocksByName.crafting_table.id,
        maxDistance: 32,
    });
    const noCraftingTableList = [
        "crafting_table", 
        "melon_seeds", "pumpkin_seeds", "sugar", 
        "iron_nugget", "iron_trapdoor", "heavy_weighted_pressure_plate", "light_weighted_pressure_plate",
        "stick", "torch", "flint_and_steel", "lever",
        "oak_planks", "birch_planks", "spruce_planks", "jungle_planks", "acacia_planks", "dark_oak_planks", "mangrove_planks",
        "cut_sandstone"];
    if (noCraftingTableList.includes(name)) {
        const recipes = bot.recipesFor(itemByName.id, null, 1, null);
        const recipe = findCraftableRecipe(bot, recipes, count);
        if (!recipe) {
            throw craftCountError(bot, name, recipes, count);
        }
        try {
            await bot.craft(recipe, count, null);
            bot.chat(`I did the recipe for ${name} ${count} times`);
            return;
        } catch (err) {
            bot.chat(`I cannot do the recipe for ${name} ${count} times`);
        }
    }
    if (!craftingTable) {
        bot.chat("Craft without a crafting table");
    } else {
        await bot.pathfinder.goto(
            new GoalLookAtBlock(craftingTable.position, bot.world)
        );
    }
    const recipes = bot.recipesFor(itemByName.id, null, 1, craftingTable);
    const recipe = findCraftableRecipe(bot, recipes, count);
    if (recipe) {
        bot.chat(`I can make ${name}`);
        try {
            await bot.craft(recipe, count, craftingTable);
            bot.chat(`I did the recipe for ${name} ${count} times`);
        } catch (err) {
            bot.chat(`I cannot do the recipe for ${name} ${count} times`);
        }
    } else {
        throw craftCountError(bot, name, recipes, count);
    }
}
