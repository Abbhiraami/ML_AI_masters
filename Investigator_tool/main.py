# pip install openai
import openai
from api_key import openai_key
import json
import pandas as pd
openai.api_key=openai_key

def extract_summary(text):
    prompt=fin_crime_prompt() + text


    response=openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {'role':'system','content':prompt}
        ]#,
        # response_format={"type": "json_object"}
    )
    final=response['choices'][0]['message']['content']

    data = json.loads(final)

    data_f = pd.DataFrame(data)
    try:
        data=json.loads(final)
        data_f=pd.DataFrame(data)
    except (json.JSONDecodeError,IndexError):
        data_f=pd.DataFrame()
        pass
    return data_f


def fin_crime_prompt():
    return '''
    You are a financial crime investigator specifically for detecting money laundering customers. 
    There will be set of transaction per customers will be provided to you. 
    Go through the transactions of list of customers and identify if the set of transactions are suspicious or genuine.  
    Use your knowledge on money laundering pattern and use that information to summarize and flag the given customer's transaction as suspicious activity. 
    Classify the following transaction group under known AML typologies like no crime ,smurfing, mule account, layering, human trafficking, or tax evasion. 
    The expected output in the json format
    
    Output format:
    {
    "customer_id": ["cus_001",'cust_002',..],
    "crime_category": ["Smurfing","Tax evasion",..],
    "risk_indicator": ["risk_indicator_001","risk_indicator_002",..],
    "summary": ["Write a detailed summary in bullets deriving the risk indicator why this crime category was assigned what made you take this decision",..]
    ...
    
    }
    
    
    =======
    Transaction Details
    '''


# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    text='''
    [
  {
    "customer_id": "CUST001",
    "transactions": [
      {"originator_country": "India", "beneficiary_country": "India", "amount_originated": 2400, "device_id": "D001", "timestamp": "04 07 2025 09:12:44"},
      {"originator_country": "India", "beneficiary_country": "UAE", "amount_originated": 9500, "device_id": "D002", "timestamp": "04 07 2025 22:45:12"},
      {"originator_country": "India", "beneficiary_country": "UAE", "amount_originated": 9700, "device_id": "D002", "timestamp": "05 07 2025 00:05:21"},
      {"originator_country": "India", "beneficiary_country": "Singapore", "amount_originated": 10400, "device_id": "D003", "timestamp": "06 07 2025 10:50:59"},
      {"originator_country": "India", "beneficiary_country": "UAE", "amount_originated": 9999, "device_id": "D002", "timestamp": "07 07 2025 01:32:18"}
    ]


    "customer_id": "CUST002",
    "transactions": [
      {"originator_country": "India", "beneficiary_country": "India", "amount_originated": 5600, "device_id": "D001", "timestamp": "03 07 2025 13:10:07"},
      {"originator_country": "India", "beneficiary_country": "USA", "amount_originated": 10200, "device_id": "D001", "timestamp": "03 07 2025 15:00:00"},
      {"originator_country": "India", "beneficiary_country": "India", "amount_originated": 3000, "device_id": "D004", "timestamp": "04 07 2025 09:30:10"},
      {"originator_country": "India", "beneficiary_country": "India", "amount_originated": 3800, "device_id": "D004", "timestamp": "06 07 2025 12:45:25"},
      {"originator_country": "India", "beneficiary_country": "India", "amount_originated": 4000, "device_id": "D001", "timestamp": "08 07 2025 17:29:41"}
    ],
  
    "customer_id": "CUST003",
    "transactions": [
      {"originator_country": "India", "beneficiary_country": "Panama", "amount_originated": 9999, "device_id": "D005", "timestamp": "05 07 2025 01:15:00"},
      {"originator_country": "India", "beneficiary_country": "Panama", "amount_originated": 9999, "device_id": "D005", "timestamp": "05 07 2025 01:18:22"},
      {"originator_country": "India", "beneficiary_country": "Panama", "amount_originated": 9999, "device_id": "D005", "timestamp": "05 07 2025 01:22:01"},
      {"originator_country": "India", "beneficiary_country": "Panama", "amount_originated": 9999, "device_id": "D006", "timestamp": "05 07 2025 01:25:30"},
      {"originator_country": "India", "beneficiary_country": "Panama", "amount_originated": 9999, "device_id": "D006", "timestamp": "05 07 2025 01:29:44"}
    ],
  
    "customer_id": "CUST004",
    "transactions": [
      {"originator_country": "India", "beneficiary_country": "India", "amount_originated": 1500, "device_id": "D007", "timestamp": "04 07 2025 08:12:12"},
      {"originator_country": "India", "beneficiary_country": "India", "amount_originated": 1700, "device_id": "D007", "timestamp": "04 07 2025 11:45:31"},
      {"originator_country": "India", "beneficiary_country": "India", "amount_originated": 1600, "device_id": "D008", "timestamp": "05 07 2025 10:03:45"},
      {"originator_country": "India", "beneficiary_country": "India", "amount_originated": 1800, "device_id": "D008", "timestamp": "06 07 2025 09:30:11"},
      {"originator_country": "India", "beneficiary_country": "India", "amount_originated": 1900, "device_id": "D007", "timestamp": "07 07 2025 13:55:22"}
    ],
 
    "customer_id": "CUST005",
    "transactions": [
      {"originator_country": "India", "beneficiary_country": "UK", "amount_originated": 11200, "device_id": "D009", "timestamp": "03 07 2025 14:20:14"},
      {"originator_country": "India", "beneficiary_country": "UK", "amount_originated": 11500, "device_id": "D009", "timestamp": "04 07 2025 09:00:11"},
      {"originator_country": "India", "beneficiary_country": "UK", "amount_originated": 11700, "device_id": "D009", "timestamp": "06 07 2025 11:44:00"},
      {"originator_country": "India", "beneficiary_country": "USA", "amount_originated": 10800, "device_id": "D005", "timestamp": "07 07 2025 13:10:05"},
      {"originator_country": "India", "beneficiary_country": "USA", "amount_originated": 11000, "device_id": "D009", "timestamp": "08 07 2025 10:27:50"}
    ]
  }
]

    '''
    df=extract_summary(text)
    print(df.to_string())


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
