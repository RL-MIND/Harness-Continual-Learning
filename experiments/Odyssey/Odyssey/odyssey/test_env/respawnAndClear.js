async function respawnAndClear(bot, spawnX = null, spawnZ = null, yaw = 0, pitch = 0, spawnY = null) {
    // A fixed X/Z makes independent same-seed worlds start from the same
    // terrain location. If Y is omitted, resolve the motion-blocking surface
    // at that X/Z and stand one block above it. Keeping spreadplayers as the
    // no-argument legacy behavior avoids changing unrelated experiment entry
    // points that explicitly want a random start.
    if (Number.isFinite(spawnX) && Number.isFinite(spawnZ)) {
        if (Number.isFinite(spawnY)) {
            await bot.chat(`/tp @s ${spawnX} ${spawnY} ${spawnZ} ${yaw} ${pitch}`);
        } else {
            await bot.chat(
                `/execute as @s positioned ${spawnX} 0 ${spawnZ} positioned over motion_blocking_no_leaves run tp @s ~ ~1 ~ ${yaw} ${pitch}`
            );
        }
    } else {
        await bot.chat("/spreadplayers ~ ~ 0 300 under 80 false @s");
        await bot.waitForTicks(20);
        await bot.chat("/execute as @s at @s positioned over motion_blocking_no_leaves run tp @s ~ ~1 ~");
    }
    await bot.waitForTicks(10);
    // Keep later deaths from introducing a second source of spawn variance.
    await bot.chat("/spawnpoint @s ~ ~ ~");
    await bot.chat("/clear @s");
    await bot.chat("/gamemode survival @s");
    await bot.chat("/difficulty peaceful");
    await bot.chat("Reset bot location and inventory.");
}
