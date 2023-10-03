"""ContextRecognition_Preprocessing - BETA VERSION"""


from pandas.core.internals.blocks import ensure_wrapped_if_datetimelike
import datetime
import math
import pytz

"""*********************************FUNCTION TO CONVERT TIMESTAMP IN DATETIME*******************"""
def timestampConverter(timestamp):
  if (timestamp == '') or (math.isnan(timestamp)):
    return ''
  else:
    # Timezone di riferimento
    timezone = pytz.timezone('America/Los_Angeles')

    # Convertire il timestamp in datetime
    utc_dt = datetime.datetime.utcfromtimestamp(timestamp)
    utc_dt = utc_dt.replace(tzinfo=pytz.utc)

    # Convertire l'UTC datetime nella timezone specificata
    local_dt = utc_dt.astimezone(timezone)

    # Formattare la data e l'ora in un formato leggibile
    formatted_datetime = local_dt.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_datetime

"""*********************************FUNCTION Context Attribute Hours*******************"""

def hourConverter(timestamp):
  if (timestamp == '') or (math.isnan(timestamp)):
    return ''
  else:
    # Timezone di riferimento
    timezone = pytz.timezone('America/Los_Angeles')

    # Convertire il timestamp in datetime
    utc_dt = datetime.datetime.utcfromtimestamp(timestamp)
    utc_dt = utc_dt.replace(tzinfo=pytz.utc)

    # Convertire l'UTC datetime nella timezone specificata
    local_dt = utc_dt.astimezone(timezone)

    # Formattare la data e l'ora in un formato leggibile
    formatted_datetime = local_dt.strftime('%-H')
    formatted_datetime = int(formatted_datetime)/24
    return formatted_datetime


"""*********************************FUNCTION Context Attribute WeekDay*******************"""
def weekDayConverter(timestamp):
  if (timestamp == '') or (math.isnan(timestamp)):
    return ''
  else:
    # Timezone
    timezone = pytz.timezone('America/Los_Angeles')

    # Convert timestamp in datetime
    utc_dt = datetime.datetime.utcfromtimestamp(timestamp)
    utc_dt = utc_dt.replace(tzinfo=pytz.utc)

    # Convert UTC datetime in timezone
    local_dt = utc_dt.astimezone(timezone)

    # Format date and time
    formatted_datetime = local_dt.strftime('%w')
    weekday = int(formatted_datetime)/7
    return weekday

"""*********************************FUNCTION Context Attribute Minutes*******************"""
def minutesConverter(timestamp):
  if (timestamp == '') or (math.isnan(timestamp)):
    return ''
  else:
    # Timezone 
    timezone = pytz.timezone('America/Los_Angeles')

    # Convert timestamp in datetime
    utc_dt = datetime.datetime.utcfromtimestamp(timestamp)
    utc_dt = utc_dt.replace(tzinfo=pytz.utc)

    # Convert UTC datetime in timezone
    local_dt = utc_dt.astimezone(timezone)

    # Format date and time
    formatted_datetime = local_dt.strftime('%M')
    min = int(formatted_datetime)/60
    return min


"""*********************************FUNCTION Context Attribute Weekend*******************"""
def isWeekend(weekday):
  if int(weekday)>0 and int(weekday) <6:
    return 0
  else:
    return 1

"""*********************************FUNCTION Context Attribute Location - Diameter *******************"""
def locationConverter(diameter):
  d = float(diameter)
  result = 0
  if d<=0:
    result =  0
  if d>0 and d <= 3.75:
    result = 0.5
  if d>3.75 and d <= 7.5:
    result = 1
  if d > 7.5 and d <= 10:
    result =1.5
  if d > 10 and d <= 15:
    result = 2
  if d > 15 and d <= 18:
    result = 2.5
  if d > 18 and d <= 22:
    result = 3
  if d > 22 and d <= 30:
    result = 4
  if d > 30 and d <= 37:
    result = 5
  if d > 37 and d <= 45:
    result = 6
  if d > 45 and d <= 52:
    result = 7
  if d > 52 and d <= 67:
    result = 8
  if d > 67 and d <= 75:
    result = 9
  if d > 75:
    result = 10
  if result > 0:
    result = result / 10

  return result

