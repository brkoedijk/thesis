from entsoe import EntsoePandasClient
import pandas as pd
API_KEY= 'Yes'
client = EntsoePandasClient(api_key=API_KEY)

start_deatlu = pd.Timestamp('20150101')
end_deatlu = pd.Timestmap('20171201')

country_code_early = 'DE_AT_LU'

ts_load_early = client.query_load(country_code_early, start=start_deatlu, end=end_deatlu)
ts_generation_early = client.query_generation(country_code_early, start=start_deatlu, end=end_deatlu, psr_type=None)


start_delu = pd.Timestamp('20180101')
end_delu = pd.Timestamp('20211231')
country_code = 'DE_LU'

ts_load = client.query_load(country_code, start=start_delu, end=end_delu)
ts_generation = client.query_generation(country_code, start=start_delu, end=end_delu, psr_type=None)






