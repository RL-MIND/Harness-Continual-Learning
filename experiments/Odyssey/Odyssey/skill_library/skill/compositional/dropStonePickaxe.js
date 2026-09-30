async function dropStonePickaxe(bot) {
  const stonePickaxe = bot.inventory.findInventoryItem(mcData.itemsByName.stone_pickaxe.id);
  if (!stonePickaxe) {
    bot.chat("No stone pickaxe found in inventory.");
    return;
  }
  await bot.tossStack(stonePickaxe);
  bot.setControlState("back", true);
  await bot.waitForTicks(25);
  bot.setControlState("back", false);
  await bot.waitForTicks(10);
  bot.chat("Dropped a stone pickaxe.");
}
