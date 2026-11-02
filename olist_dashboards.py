"""Native Olist dashboards. Call render_dashboards() after your login gate."""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

SPECS = {
 'customers': ['customer_id','customer_unique_id','customer_state','customer_city','customer_zip_code_prefix'],
 'orders': ['order_id','customer_id','order_status','order_purchase_timestamp','order_delivered_customer_date','order_estimated_delivery_date'],
 'items': ['order_id','order_item_id','product_id','seller_id','price','freight_value'],
 'payments': ['order_id','payment_sequential','payment_type','payment_value','payment_installments'],
 'reviews': ['order_id','review_score'],
 'products': ['product_id','product_category_name','product_weight_g','product_photos_qty'],
 'sellers': ['seller_id','seller_state','seller_city'],
 'geolocation': ['geolocation_zip_code_prefix','geolocation_lat','geolocation_lng'],
 'categories': ['product_category_name','product_category_name_english'],
}
KEYS = {'customers':['customer_id'],'orders':['order_id'],'items':['order_id','order_item_id'],
        'payments':['order_id','payment_sequential'],'products':['product_id'],
        'sellers':['seller_id'],'categories':['product_category_name']}

@st.cache_data(show_spinner=False, max_entries=2)
def load_data(signature):
    tables = {}
    for filename, _, _ in signature:
        path = Path(filename)
        header = pd.read_csv(path, nrows=0).columns.str.strip().str.lower()
        candidates = [name for name, cols in SPECS.items() if set(cols).issubset(header)]
        if not candidates:
            continue
        if len(candidates) != 1:
            raise ValueError(f'Ambiguous table schema: {path.name}')
        name = candidates[0]
        if name in tables:
            raise ValueError(f'Two CSVs match {name}. Keep only one current export of each table.')
        # Strings preserve leading zero ZIP prefixes and identifiers.
        df = pd.read_csv(path, dtype=str, low_memory=False)
        df.columns = df.columns.str.strip().str.lower()
        df = df.replace(r'^\s*$', np.nan, regex=True)
        if name in KEYS:
            key = KEYS[name]
            if df[key].isna().any(axis=1).any() or df.duplicated(key).any():
                raise ValueError(f'{path.name}: missing or duplicate keys {key}. Fix the cleaned export first.')
        tables[name] = df
    missing = set(SPECS) - set(tables)
    if missing:
        raise ValueError('Missing tables or required columns: ' + ', '.join(sorted(missing)) +
                         '. See DATA_COLUMNS.md for the required schemas.')
    for name, cols in {
        'items':['price','freight_value'],
        'payments':['payment_value','payment_installments'],
        'reviews':['review_score'],
        'products':['product_weight_g','product_photos_qty'],
        'geolocation':['geolocation_lat','geolocation_lng'],
    }.items():
        for col in cols:
            tables[name][col] = pd.to_numeric(tables[name][col], errors='coerce')
    for col in ['order_purchase_timestamp','order_delivered_customer_date','order_estimated_delivery_date']:
        tables['orders'][col] = pd.to_datetime(tables['orders'][col], errors='coerce')
    for name, col in [('customers','customer_zip_code_prefix'),('geolocation','geolocation_zip_code_prefix')]:
        tables[name][col] = tables[name][col].str.replace(r'\.0$', '', regex=True).str.zfill(5)
    return tables

def assemble(t):
    """One row per order; never join raw payments and items together."""
    i = t['items'].groupby('order_id', as_index=False).agg(
        item_gmv=('price',lambda s:s.sum(min_count=1)),
        freight=('freight_value',lambda s:s.sum(min_count=1)), item_count=('order_item_id','size'))
    p = t['payments'].groupby('order_id',as_index=False).agg(
        payment_total=('payment_value',lambda s:s.sum(min_count=1)))
    valid = t['reviews'].loc[t['reviews'].review_score.between(1,5)]
    r = valid.groupby('order_id',as_index=False).agg(mean_review=('review_score','mean'))
    orders = t['orders'].merge(t['customers'],on='customer_id',how='left',validate='many_to_one')
    for part in [i,p,r]:
        orders = orders.merge(part,on='order_id',how='left',validate='one_to_one')
    orders['delivery_days'] = (orders.order_delivered_customer_date-orders.order_purchase_timestamp).dt.total_seconds()/86400
    orders.loc[orders.delivery_days.lt(0),'delivery_days'] = np.nan
    eligible = orders.order_status.eq('delivered') & orders.order_delivered_customer_date.notna() & orders.order_estimated_delivery_date.notna()
    orders['late'] = np.nan
    orders.loc[eligible,'late'] = (orders.loc[eligible,'order_delivered_customer_date'].dt.normalize() > orders.loc[eligible,'order_estimated_delivery_date'].dt.normalize()).astype(int)
    ip = t['items'].merge(t['products'],on='product_id',how='left',validate='many_to_one')
    ip = ip.merge(t['categories'],on='product_category_name',how='left',validate='many_to_one')
    ip['category'] = ip.product_category_name_english.fillna(ip.product_category_name).fillna('Unknown')
    return orders, ip

def metrics(values):
    for box, (label,value) in zip(st.columns(len(values)), values):
        box.metric(label,value)

def number(x):
    return 'N/A' if pd.isna(x) else f'{x:,.0f}'

def money(x):
    return 'N/A' if pd.isna(x) else f'R$ {x:,.2f}'

def percent(x):
    return 'N/A' if pd.isna(x) else f'{x:.1%}'

def bars(series,title,limit=15):
    st.subheader(title)
    series = series.dropna().sort_values(ascending=False).head(limit)
    if series.empty:
        st.info('No values for this selection.')
    else:
        st.bar_chart(series.rename('Value'))

def counts(df,col,title):
    bars(df[col].fillna('Unknown').value_counts(),title)

def monthly(df,value,title):
    st.subheader(title)
    x = df.dropna(subset=['order_purchase_timestamp']).copy()
    if x.empty:
        st.info('No dated orders for this selection.'); return
    x['Month'] = x.order_purchase_timestamp.dt.to_period('M').astype(str)
    s = x.groupby('Month')[value].sum(min_count=1) if value else x.groupby('Month').size()
    st.line_chart(s.rename(value or 'Orders'))

def choose(label,series,key):
    return st.multiselect(label, sorted(series.dropna().astype(str).unique().tolist()),key=key)

def render_dashboards():
    st.title('Olist Business Dashboards')
    logo = Path(__file__).with_name('olist_logo.png')
    if logo.exists(): st.image(str(logo),width=110)
    st.caption('Historical analysis · Brazilian real (R$) · purchase-date filters')
    root = Path(__file__).resolve().parent / 'data'
    paths = sorted(root.glob('*.csv')) if root.exists() else []
    if not paths:
        st.info('Create a data folder beside app.py and copy your nine cleaned CSV exports into it.')
        st.code(str(root)); return
    signature = tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in paths)
    try:
        t = load_data(signature)
        orders, all_items = assemble(t)
    except (ValueError,KeyError,pd.errors.ParserError) as error:
        st.error(str(error)); return
    page = st.selectbox('Dashboard page',['Customers','Orders','Order Items','Payments','Geolocation','Reviews','Products','Sellers','Categories'])
    with st.expander('Filters',expanded=True):
        a,b,c = st.columns(3)
        with a: states = choose('Customer state',orders.customer_state,'dash_state')
        with b: statuses = choose('Order status',orders.order_status,'dash_status')
        with c: methods = choose('Payment method (orders using any selected method)',t['payments'].payment_type,'dash_payment')
        cats = choose('Category (orders containing any selected category)',all_items.category,'dash_category')
        dated = orders.order_purchase_timestamp.dropna()
        dates = None
        if not dated.empty:
            dates = st.date_input('Purchase date range',(dated.min().date(),dated.max().date()),
                                  min_value=dated.min().date(),max_value=dated.max().date(),key='dash_dates')
    selected = orders.copy()
    if states: selected = selected[selected.customer_state.isin(states)]
    if statuses: selected = selected[selected.order_status.isin(statuses)]
    if methods:
        ids = t['payments'].loc[t['payments'].payment_type.isin(methods),'order_id']
        selected = selected[selected.order_id.isin(ids)]
    if cats:
        ids = all_items.loc[all_items.category.isin(cats),'order_id']
        selected = selected[selected.order_id.isin(ids)]
    if dates is not None:
        if len(dates) != 2:
            st.info('Select both the start and end date.'); return
        selected = selected[selected.order_purchase_timestamp.ge(pd.Timestamp(dates[0])) & selected.order_purchase_timestamp.lt(pd.Timestamp(dates[1])+pd.Timedelta(days=1))]
    st.caption('Blank multi-select = all. Category and payment filters select whole orders; all items and payments of matching orders remain included. Undated orders are excluded by the date range.')
    if selected.empty:
        st.info('No orders match these filters.'); return
    ids = selected.order_id
    items = all_items[all_items.order_id.isin(ids)]
    pay = t['payments'][t['payments'].order_id.isin(ids)]
    review = t['reviews'][t['reviews'].order_id.isin(ids) & t['reviews'].review_score.between(1,5)]
    products = t['products'][t['products'].product_id.isin(items.product_id)]
    sellers = t['sellers'][t['sellers'].seller_id.isin(items.seller_id)]
    if page == 'Customers':
        per_customer = selected.dropna(subset=['customer_unique_id']).groupby('customer_unique_id').order_id.nunique()
        metrics([('Unique customers',number(per_customer.size)),('Repeat customer share',percent(per_customer.gt(1).mean())),('Item GMV',money(selected.item_gmv.sum(min_count=1))),('Mean order review',number(selected.mean_review.mean()) if selected.mean_review.notna().sum()==0 else f'{selected.mean_review.mean():.2f}')])
        bars(selected.groupby('customer_state').customer_unique_id.nunique(),'Customers by state')
        monthly(selected,None,'Orders by month')
        counts(selected,'customer_city','Top customer cities (order count)')
        st.caption('Repeat share: customers with more than one selected order / selected unique customers. It is not lifetime repeat behavior.')
    elif page == 'Orders':
        metrics([('Orders',number(len(selected))),('Item GMV',money(selected.item_gmv.sum(min_count=1))),('Mean delivery days',f'{selected.delivery_days.mean():.1f}' if selected.delivery_days.notna().any() else 'N/A'),('Late delivery share',percent(selected.late.mean()))])
        monthly(selected,None,'Orders by month'); counts(selected,'order_status','Order status')
        bars(selected.groupby('customer_state').item_gmv.sum(min_count=1),'Item GMV by customer state')
        st.caption('Late share denominator: delivered orders with both delivery and estimated dates. A delivery later on the same calendar day is not late.')
    elif page == 'Order Items':
        metrics([('Item rows',number(len(items))),('Products sold',number(items.product_id.nunique())),('Item GMV',money(items.price.sum(min_count=1))),('Freight',money(items.freight_value.sum(min_count=1)))])
        bars(items.groupby('category').price.sum(min_count=1),'Item GMV by category')
        bars(selected.item_count.value_counts().sort_index(),'Orders by number of items',30)
        bars(items.groupby('seller_id').price.sum(min_count=1),'Top sellers by item GMV')
    elif page == 'Payments':
        metrics([('Payment records',number(len(pay))),('Payment total',money(pay.payment_value.sum(min_count=1))),('Mean record value',money(pay.payment_value.mean())),('Mean installments per record',f'{pay.payment_installments.mean():.2f}' if pay.payment_installments.notna().any() else 'N/A')])
        bars(pay.groupby('payment_type').payment_value.sum(min_count=1),'Payment value by method')
        counts(pay,'payment_installments','Payment records by installments')
        monthly(selected,'payment_total','Payment value by purchase month')
    elif page == 'Reviews':
        metrics([('Valid review records',number(len(review))),('Reviewed orders',number(review.order_id.nunique())),('Mean order rating',f'{selected.mean_review.mean():.2f}' if selected.mean_review.notna().any() else 'N/A'),('Low satisfaction share',percent(selected.loc[selected.mean_review.notna(),'mean_review'].lt(3).mean()))])
        counts(review,'review_score','Review records by star rating')
        bars(selected.groupby('customer_state').mean_review.mean(),'Mean order rating by state',30)
        groups = selected.dropna(subset=['mean_review','late']).groupby('late').mean_review.mean().rename(index={0.0:'On time',1.0:'Late'})
        bars(groups,'Mean order rating: on time vs late')
        st.caption('Valid scores are 1–5. Multiple reviews are averaged per order for the low-satisfaction KPI; star counts remain review-record counts.')
    elif page == 'Products':
        metrics([('Products sold',number(len(products))),('Full catalog products',number(len(t['products']))),('Mean listed weight (g)',number(products.product_weight_g.mean())),('Mean listed photo count',f'{products.product_photos_qty.mean():.1f}' if products.product_photos_qty.notna().any() else 'N/A')])
        bars(items.groupby('product_id').price.sum(min_count=1),'Top products by item GMV')
        bars(items.groupby('category').product_id.nunique(),'Products sold by category')
        counts(products,'product_photos_qty','Sold products by photo count')
        st.caption('Product attributes are averaged once per sold product; full catalog count ignores order filters.')
    elif page == 'Sellers':
        metrics([('Active sellers',number(len(sellers))),('Full seller catalog',number(len(t['sellers']))),('Seller states',number(sellers.seller_state.nunique())),('Item GMV',money(items.price.sum(min_count=1)))])
        counts(sellers,'seller_state','Active sellers by state')
        bars(items.groupby('seller_id').price.sum(min_count=1),'Top sellers by item GMV')
        counts(sellers,'seller_city','Active sellers by city')
    elif page == 'Categories':
        sold = items.drop_duplicates('product_id')
        coverage = sold.product_category_name_english.notna().mean()
        metrics([('Sold categories',number(items.category.nunique())),('Products sold',number(len(sold))),('Translation coverage (sold products)',percent(coverage)),('Untranslated sold products',number(sold.product_category_name_english.isna().sum()))])
        bars(items.groupby('category').price.sum(min_count=1),'Category item GMV')
        bars(items.groupby('category').order_id.nunique(),'Orders containing each category')
        st.caption('Multi-category orders count in each category. Category order counts are not additive.')
    elif page == 'Geolocation':
        customers = selected.drop_duplicates('customer_id')
        geo = t['geolocation'].copy()
        good = geo.geolocation_lat.between(-90,90) & geo.geolocation_lng.between(-180,180)
        geo = geo.loc[good].groupby('geolocation_zip_code_prefix',as_index=False).agg(lat=('geolocation_lat','median'),lon=('geolocation_lng','median'))
        points = customers[['customer_zip_code_prefix']].drop_duplicates().merge(geo,left_on='customer_zip_code_prefix',right_on='geolocation_zip_code_prefix',how='left',validate='one_to_one')
        metrics([('Customer ZIP prefixes',number(len(points))),('Mapped ZIP prefixes',number(points.lat.notna().sum())),('ZIP coverage',percent(points.lat.notna().mean())),('Customer states',number(customers.customer_state.nunique()))])
        mapped = points.dropna(subset=['lat','lon'])
        if not mapped.empty: st.map(mapped[['lat','lon']])
        else: st.info('No valid coordinates for the selected customer ZIP prefixes.')
        counts(selected,'customer_state','Orders by customer state')
        st.caption('One approximate median coordinate per ZIP prefix; points represent ZIP prefixes, not individual customers. The map requires internet access.')
    with st.expander('Metric definitions and filtered order export'):
        st.write('Item GMV = sum of item prices, excluding freight; it is not net revenue. Payment total is separate and can include freight. No currency conversion is applied. No automatic DAX parity is claimed.')
        st.download_button('Download filtered order summary',selected.to_csv(index=False).encode('utf-8'),file_name='olist_filtered_orders.csv',mime='text/csv')
