async function placeItem(bot, name, position) {
    // return if name is not string
    if (typeof name !== "string") {
        throw new Error(`name for placeItem must be a string`);
    }
    // return if position is not Vec3
    if (!(position instanceof Vec3)) {
        throw new Error(`position for placeItem must be a Vec3`);
    }
    const itemByName = mcData.itemsByName[name];
    if (!itemByName) {
        throw new Error(`No item named ${name}`);
    }
    const item = bot.inventory.findInventoryItem(itemByName.id);
    if (!item) {
        bot.chat(`No ${name} in inventory`);
        return;
    }
    // Count all stacks, rather than only the stack selected for equip. This is
    // the inventory invariant used to verify a consumptive placement.
    const countBefore = bot.inventory.count(itemByName.id);
    // find a reference block
    const faceVectors = [
        new Vec3(0, 1, 0),
        new Vec3(0, -1, 0),
        new Vec3(1, 0, 0),
        new Vec3(-1, 0, 0),
        new Vec3(0, 0, 1),
        new Vec3(0, 0, -1),
    ];
    let referenceBlock = null;
    let faceVector = null;
    for (const vector of faceVectors) {
        const block = bot.blockAt(position.minus(vector));
        if (block?.name !== "air") {
            referenceBlock = block;
            faceVector = vector;
            bot.chat(`Placing ${name} on ${block.name} at ${block.position}`);
            break;
        }
    }
    if (!referenceBlock) {
        bot.chat(
            `No block to place ${name} on. You cannot place a floating block.`
        );
        _placeItemFailCount++;
        if (_placeItemFailCount > 10) {
            throw new Error(
                `placeItem failed too many times. You cannot place a floating block.`
            );
        }
        return;
    }

    const placementVerified = () => {
        const placedBlock = bot.blockAt(position);
        const countAfter = bot.inventory.count(itemByName.id);
        return {
            blockPlaced: placedBlock?.name === name,
            countAfter,
            inventoryConsumed: countAfter === countBefore - 1,
        };
    };
    const waitForPlacementSync = async (maxTicks = 40, pollTicks = 2) => {
        let verification = placementVerified();
        let waitedTicks = 0;
        while (
            (!verification.blockPlaced || !verification.inventoryConsumed) &&
            waitedTicks < maxTicks &&
            typeof bot.waitForTicks === "function"
        ) {
            await bot.waitForTicks(pollTicks);
            waitedTicks += pollTicks;
            verification = placementVerified();
        }
        return verification;
    };

    // A chat message is not proof of success. Verify both the world state and
    // inventory state so callers can distinguish an actual placement from a
    // server/observation desynchronization.
    try {
        // You must first go to the block position you want to place
        await bot.pathfinder.goto(new GoalPlaceBlock(position, bot.world, {}));
        // You must equip the item right before calling placeBlock
        await bot.equip(item, "hand");
        await bot.placeBlock(referenceBlock, faceVector);
        const verification = await waitForPlacementSync();
        if (!verification.blockPlaced || !verification.inventoryConsumed) {
            throw new Error(
                `placement_inventory_desync: item=${name} before=${countBefore} after=${verification.countAfter} blockPlaced=${verification.blockPlaced}`
            );
        }
        bot.chat(`Placed ${name}`);
        bot.save(`${name}_placed`);
    } catch (err) {
        // Mineflayer can occasionally reject after the server already applied
        // the placement. Treat that as success only after the same two-part
        // verification, never based on a changed hand slot or chat message.
        const verification = await waitForPlacementSync();
        if (verification.blockPlaced && verification.inventoryConsumed) {
            bot.chat(`Placed ${name}`);
            bot.save(`${name}_placed`);
            return;
        }
        _placeItemFailCount++;
        const detail = err?.message || String(err);
        bot.chat(`Error placing ${name}: ${detail}`);
        throw new Error(
            `placement_failed: item=${name} before=${countBefore} after=${verification.countAfter} blockPlaced=${verification.blockPlaced} cause=${detail}`
        );
    }
}
