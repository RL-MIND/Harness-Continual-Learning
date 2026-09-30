async function equipItem(bot, itemName) {
    // This primitive deliberately has one responsibility: select an item that
    // already exists in the inventory and put it in the main hand.  Do not add
    // recovery behavior here (crafting, placing, mining, dropping, etc.);
    // callers can observe a failed precondition and plan that work explicitly.
    if (typeof itemName !== "string" || itemName.length === 0) {
        throw new Error("itemName for equipItem must be a non-empty string");
    }
    if (!mcData.itemsByName[itemName]) {
        throw new Error(`No item named ${itemName}`);
    }

    const item = bot.inventory.items().find((candidate) => candidate.name === itemName);
    if (!item) {
        const result = { ok: false, itemName, reason: "item_not_in_inventory" };
        bot.chat(`Cannot equip ${itemName}: it is not in inventory.`);
        return result;
    }

    const before = bot.heldItem ? bot.heldItem.name : null;
    await bot.equip(item, "hand");
    const after = bot.heldItem ? bot.heldItem.name : null;
    if (after !== itemName) {
        throw new Error(`equipItem verification failed: expected ${itemName}, found ${after || "empty hand"}`);
    }

    const result = { ok: true, itemName, before, after, evidence: "main_hand" };
    bot.chat(`Equipped ${itemName}.`);
    return result;
}
