"""
SQL query
"""

MATCHED_QUERY = """
  select m.datetime, m.tickersymbol, m.price
  from quote.matched m
  join quote.futurecontractcode fc on date(m.datetime) = fc.datetime and fc.tickersymbol = m.tickersymbol
  where fc.futurecode = %s and m.datetime between %s and %s and
        ((EXTRACT(HOUR FROM m.datetime) >= 9 AND EXTRACT(HOUR FROM m.datetime) < 14)
        OR (EXTRACT(HOUR FROM m.datetime) = 14 AND EXTRACT(MINUTE FROM m.datetime) <= 30))
  order by m.datetime
"""

BID_ASK_QUERY = """
  select b.datetime, b.tickersymbol, b.price, a.price, a.price - b.price
  from quote.bidprice b join quote.askprice a
  on b.datetime = a.datetime and b.tickersymbol = a.tickersymbol and b.depth = a.depth
  join quote.futurecontractcode fc on date(b.datetime) = fc.datetime and fc.tickersymbol = b.tickersymbol
  where b.depth = 1 and fc.futurecode = %s and b.datetime between %s and %s and
        ((EXTRACT(HOUR FROM b.datetime) >= 9 AND EXTRACT(HOUR FROM b.datetime) < 14)
        OR (EXTRACT(HOUR FROM b.datetime) = 14 AND EXTRACT(MINUTE FROM b.datetime) <= 30))
  order by b.datetime
"""

CLOSE_QUERY = """
  select c.datetime, c.tickersymbol, c.price
  from quote.close c
  join quote.futurecontractcode fc on c.datetime = fc.datetime and fc.tickersymbol = c.tickersymbol
  where fc.futurecode = %s and c.datetime between %s and %s
  order by c.datetime
"""
ASK_SIZE_QUERY = """
  select m.datetime, m.tickersymbol, m.quantity, m.depth
  from quote.asksize m
  join quote.futurecontractcode fc on date(m.datetime) = fc.datetime and fc.tickersymbol = m.tickersymbol
  where fc.futurecode = %s and m.datetime between %s and %s and
        ((EXTRACT(HOUR FROM m.datetime) >= 9 AND EXTRACT(HOUR FROM m.datetime) < 14)
        OR (EXTRACT(HOUR FROM m.datetime) = 14 AND EXTRACT(MINUTE FROM m.datetime) <= 30))
  order by m.datetime
"""
MATCHED_VOLUME_QUERY = """
  select m.datetime, m.tickersymbol, m.quantity
  from quote.matchedvolume m
  join quote.futurecontractcode fc on date(m.datetime) = fc.datetime and fc.tickersymbol = m.tickersymbol
  where fc.futurecode = %s and m.datetime between %s and %s and
        ((EXTRACT(HOUR FROM m.datetime) >= 9 AND EXTRACT(HOUR FROM m.datetime) < 14)
        OR (EXTRACT(HOUR FROM m.datetime) = 14 AND EXTRACT(MINUTE FROM m.datetime) <= 30))
  order by m.datetime
"""
BID_SIZE_QUERY = """
  select m.datetime, m.tickersymbol, m.quantity, m.depth
  from quote.bidsize m
  join quote.futurecontractcode fc on date(m.datetime) = fc.datetime and fc.tickersymbol = m.tickersymbol
  where fc.futurecode = %s and m.datetime between %s and %s and
        ((EXTRACT(HOUR FROM m.datetime) >= 9 AND EXTRACT(HOUR FROM m.datetime) < 14)
        OR (EXTRACT(HOUR FROM m.datetime) = 14 AND EXTRACT(MINUTE FROM m.datetime) <= 30))
  order by m.datetime
"""