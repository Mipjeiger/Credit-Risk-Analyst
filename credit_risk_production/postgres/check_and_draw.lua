-- credit_risk_production/postgres/check_and_draw.lua
-- KEYS[1] = limit key, KEYS[2] = drawn key
-- ARGV[1] = amount to draw
-- returns -1 if would exceed, otherwise the new drawn total

local limit = tonumber(redis.call("GET", KEYS[1]) or "0")
local drawn = tonumber(redis.call("GET", KEYS[2]) or "0")
local amount = tonumber(ARGV[1])

if drawn + amount > limit then
    return -1
end

redis.call('INCRBYFLOAT', KEYS[2], amount)
return drawn + amount