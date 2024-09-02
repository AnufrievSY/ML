import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")
ROOT = os.getcwd().split('\\LLM')[0] + '\\LLM\\data\\'
supplier_df = pd.read_pickle(os.path.join(ROOT, "supplier_df.pkl"))
category_df = pd.read_pickle(os.path.join(ROOT, "category_df.pkl"))
mapping_df = pd.read_pickle(os.path.join(ROOT, "mapping_df.pkl"))

mapping_df = mapping_df[['Артикул', 'type_id', 'description_category_id']]
mapping_df.drop_duplicates(inplace=True)

vcdf = mapping_df['Артикул'].value_counts()
vcdf = vcdf[vcdf==1]
articuls = vcdf.index.tolist()

clean_mapping_df = mapping_df[mapping_df['Артикул'].isin(articuls)]

clean_supplier_df = supplier_df[supplier_df['Код артикула'].isin(clean_mapping_df['Артикул'])]
clean_supplier_df.loc[:, 'text'] = clean_supplier_df.apply(lambda row: '/'.join(row[['Название', 'Группа товаров', 'Раздел']].tolist()), axis=1)
clean_supplier_df.rename(columns={'Код артикула':'articul'}, inplace=True)

prepare_supplier_df = clean_supplier_df[['articul', 'text']]
if prepare_supplier_df.shape[0] != clean_supplier_df.shape[0]:
  print('ERROR: не все товары были найдены у поставщика')

def get_category_text(v) -> str | None:
    type_id = v['type_id']
    description_category_id = v['description_category_id']
    filtered_category_df = category_df[((category_df['2_type_id'] == type_id) & (category_df['1_description_category_id'] == description_category_id))]
    if filtered_category_df.empty:
      print(f'NOT FOUND ERROR: {type_id}')
      return None
    elif filtered_category_df.shape[0] > 1:
      print(f'NOT UNIQUE ERROR: {type_id}')
    else:
      return '/'.join(filtered_category_df.iloc[0][['0_category_name', '1_category_name', '2_type_name']].tolist())

data = pd.DataFrame({
    'text': clean_mapping_df.merge(right=prepare_supplier_df, left_on='Артикул', right_on='articul', how='left')['text'].tolist(),
    'label': clean_mapping_df[['description_category_id', 'type_id']].apply(get_category_text, axis=1).tolist()
})

