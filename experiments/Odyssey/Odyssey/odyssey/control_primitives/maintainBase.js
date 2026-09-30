// Bind a model-designated durable block to a verified world coordinate.
async function maintainRegisteredAsset(bot, savedAnchor, assetName, savedPosition, savedRejectedPositions = []) {
    if (typeof assetName !== "string" || !/^[a-z0-9_]+$/.test(assetName)) {
        throw new Error("assetName must be a concrete Minecraft block name");
    }
    let position = normalizedPosition(savedPosition);
    let anchor = normalizedPosition(savedAnchor);
    if (!position && anchor) {
        // A prior placement may have reached the server while its inventory
        // update arrived too late for placeItem's strict verification. Inspect
        // the real world before selecting another coordinate so retries reuse
        // the existing asset instead of creating a duplicate.
        await useRegisteredAsset(bot, assetName, anchor, true);
        const existingPosition = findNearbyRegisteredAsset(bot, assetName, anchor);
        if (existingPosition) {
            bot.save(`harness_asset:${JSON.stringify({
                asset: assetName,
                anchor,
                position: existingPosition,
                status: "present",
                detail: "Recovered an existing nearby asset from world state."
            })}`);
            return;
        }
    }
    if (!position) {
        position = findRegisteredAssetPosition(bot, anchor, savedRejectedPositions);
        if (!position) {
            bot.save(`harness_asset:${JSON.stringify({ asset: assetName, anchor: null, position: null, status: "no_flat_position" })}`);
            return;
        }
    }
    if (!anchor) anchor = { x: position.x, y: position.y, z: position.z };
    await useRegisteredAsset(bot, assetName, position, true);
    const block = bot.blockAt(new Vec3(position.x, position.y, position.z));
    let status = "unknown";
    if (block?.name === assetName) {
        status = "present";
    } else {
        // The durable coordinate can become stale when an older skill moves
        // or destroys an asset. Prefer a world-observed asset of the exact
        // same type near the base anchor over repeatedly repairing an empty
        // coordinate. The report remains authoritative: Python only rebinds
        // after this concrete block observation.
        const nearbyPosition = findNearbyRegisteredAsset(bot, assetName, anchor);
        if (
            nearbyPosition &&
            (nearbyPosition.x !== position.x ||
                nearbyPosition.y !== position.y ||
                nearbyPosition.z !== position.z)
        ) {
            bot.save(`harness_asset:${JSON.stringify({
                asset: assetName,
                anchor,
                position: nearbyPosition,
                status: "present",
                detail: `Rebound stale registered coordinate from (${position.x},${position.y},${position.z}).`
            })}`);
            return;
        }
        if (block && block.name !== "air") {
            status = `conflict:${block.name}`;
            bot.save(`harness_asset:${JSON.stringify({ asset: assetName, anchor, position, status })}`);
            return;
        }
        const item = mcData.itemsByName[assetName];
        if (!item || !bot.inventory.findInventoryItem(item.id)) {
            status = "missing_inventory_item";
        } else {
            try {
                await placeItem(bot, assetName, new Vec3(position.x, position.y, position.z));
                status = bot.blockAt(new Vec3(position.x, position.y, position.z))?.name === assetName ? "placed" : "placement_unverified";
            } catch (err) {
                const detail = err?.message || String(err);
                // World state is authoritative for a persistent asset. If the
                // block exists after a placement/inventory desynchronization,
                // register it and prohibit a second placement elsewhere.
                if (bot.blockAt(new Vec3(position.x, position.y, position.z))?.name === assetName) {
                    status = "present";
                    bot.save(`harness_asset:${JSON.stringify({ asset: assetName, anchor, position, status, detail })}`);
                    return;
                }
                status = "placement_failed";
                bot.save(`harness_asset:${JSON.stringify({ asset: assetName, anchor, position, status, detail })}`);
                return;
            }
        }
    }
    bot.save(`harness_asset:${JSON.stringify({ asset: assetName, anchor, position, status })}`);
}

async function useRegisteredAsset(bot, assetName, position, allowEmpty = false) {
    const target = normalizedPosition(position);
    if (!target) throw new Error("registered asset position is invalid");
    const targetVec = new Vec3(target.x, target.y, target.z);
    const block = bot.blockAt(targetVec);
    if (!allowEmpty && block?.name !== assetName) {
        throw new Error(`registered_asset_missing: expected=${assetName} actual=${block?.name}`);
    }
    if (targetVec.distanceTo(bot.entity.position) > 8) {
        await bot.pathfinder.goto(new GoalNear(target.x, target.y, target.z, 3));
    }
}

function normalizedPosition(value) {
    if (!value || !Number.isInteger(value.x) || !Number.isInteger(value.y) || !Number.isInteger(value.z)) return null;
    return { x: value.x, y: value.y, z: value.z };
}

function findNearbyRegisteredAsset(bot, assetName, savedAnchor, maxRadius = 12) {
    const anchor = normalizedPosition(savedAnchor);
    if (!anchor) return null;
    const verticalOffsets = [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6];
    for (let radius = 0; radius <= maxRadius; radius++) {
        for (let dx = -radius; dx <= radius; dx++) {
            for (let dz = -radius; dz <= radius; dz++) {
                if (radius > 0 && Math.max(Math.abs(dx), Math.abs(dz)) !== radius) continue;
                for (const dy of verticalOffsets) {
                    const position = {
                        x: anchor.x + dx,
                        y: anchor.y + dy,
                        z: anchor.z + dz
                    };
                    if (bot.blockAt(new Vec3(position.x, position.y, position.z))?.name === assetName) {
                        return position;
                    }
                }
            }
        }
    }
    return null;
}

function findRegisteredAssetPosition(bot, savedAnchor = null, savedRejectedPositions = []) {
    const origin = normalizedPosition(savedAnchor) || bot.entity.position.floored();
    const rejected = new Set(
        (Array.isArray(savedRejectedPositions) ? savedRejectedPositions : [])
            .map(normalizedPosition)
            .filter(Boolean)
            .map((position) => `${position.x},${position.y},${position.z}`)
    );
    for (let radius = 1; radius <= 12; radius++) {
        for (let dx = -radius; dx <= radius; dx++) {
            for (let dz = -radius; dz <= radius; dz++) {
                if (Math.max(Math.abs(dx), Math.abs(dz)) !== radius) continue;
                for (const dy of [0, -1, 1]) {
                    const position = { x: origin.x + dx, y: origin.y + dy, z: origin.z + dz };
                    if (rejected.has(`${position.x},${position.y},${position.z}`)) continue;
                    const target = new Vec3(position.x, position.y, position.z);
                    const ground = bot.blockAt(target.offset(0, -1, 0));
                    const head = bot.blockAt(target.offset(0, 1, 0));
                    const block = bot.blockAt(target);
                    if (ground && ground.name !== "air" && ground.name !== "water" && ground.name !== "lava" && block?.name === "air" && head?.name === "air") {
                        return position;
                    }
                }
            }
        }
    }
    return null;
}
