// Prevent model-designated persistent assets from being destroyed as
// collateral damage by pathfinding or collection skills.
function protectRegisteredAssets(bot, assets = []) {
    const normalizedAssets = (Array.isArray(assets) ? assets : [])
        .map((asset) => {
            const position = asset?.position;
            if (
                !position ||
                !Number.isInteger(position.x) ||
                !Number.isInteger(position.y) ||
                !Number.isInteger(position.z)
            ) {
                return null;
            }
            return {
                block_name: typeof asset.block_name === "string" ? asset.block_name : "world_asset",
                position: { x: position.x, y: position.y, z: position.z },
            };
        })
        .filter(Boolean);

    const protectedByPosition = new Map(
        normalizedAssets.map((asset) => [
            `${asset.position.x},${asset.position.y},${asset.position.z}`,
            asset,
        ])
    );
    // The dig guard is installed once, while this map is replaced before
    // every Harness action so newly registered or relocated assets take
    // effect without stacking wrappers.
    bot._harnessProtectedAssets = protectedByPosition;

    const positionKey = (position) =>
        position && Number.isFinite(position.x) && Number.isFinite(position.y) && Number.isFinite(position.z)
            ? `${Math.floor(position.x)},${Math.floor(position.y)},${Math.floor(position.z)}`
            : null;

    if (!bot._harnessOriginalDig && typeof bot.dig === "function") {
        bot._harnessOriginalDig = bot.dig.bind(bot);
        bot.dig = async function guardedDig(block, ...args) {
            const key = positionKey(block?.position);
            const protectedAsset = key ? bot._harnessProtectedAssets?.get(key) : null;
            if (protectedAsset) {
                throw new Error(
                    `protected_world_asset: refusing to dig ${protectedAsset.block_name} at ${key}`
                );
            }
            return bot._harnessOriginalDig(block, ...args);
        };
    }

    const installPathProtection = (movements) => {
        if (!movements || !Array.isArray(movements.exclusionAreasBreak)) return;
        movements.exclusionAreasBreak = movements.exclusionAreasBreak.filter(
            (callback) => !callback?._harnessProtectedAssetGuard
        );
        const guard = (block) => {
            const key = positionKey(block?.position);
            return key && bot._harnessProtectedAssets?.has(key) ? 100 : 0;
        };
        guard._harnessProtectedAssetGuard = true;
        movements.exclusionAreasBreak.push(guard);
    };

    // collectBlock owns a separate Movements instance and replaces the
    // pathfinder's active instance whenever collection starts, so both must
    // carry the same coordinate guard.
    installPathProtection(bot.pathfinder?.movements);
    installPathProtection(bot.collectBlock?.movements);
    if (bot.pathfinder?.movements && typeof bot.pathfinder.setMovements === "function") {
        bot.pathfinder.setMovements(bot.pathfinder.movements);
    }
}
