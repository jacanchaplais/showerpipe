import numpy as np

def df_to_struc(df):
    def df_to_rec(df):
        return df.to_records(index=False)
    def rec_to_struc(rec):
        return rec.view(rec.dtype.fields or rec.dtype, np.ndarray)
    return rec_to_struc(df_to_rec(df))

def vertex_df(event_df):
    vertex_df = event_df.reset_index()
    vertex_df = vertex_df.pivot_table(
            index='parents',
            values=['index'],
            aggfunc=lambda x: tuple(x.to_list())
            )
    vertex_df = vertex_df.reset_index()
    vertex_df = vertex_df.rename(columns={'parents': 'in', 'index': 'out'})
    vertex_df.index.name = 'id'
    return vertex_df

def unpack(vtx_df, direction):
    vtx_col = vtx_df[direction].reset_index()
    vtx_col = vtx_col.explode(direction)
    vtx_col = vtx_col.set_index(direction)
    vertex_to_edge = {'in': 'out', 'out': 'in'}
    vtx_col.index.name = 'index'
    vtx_col = vtx_col.rename(columns={'id': vertex_to_edge[direction]})
    return vtx_col

def add_edge_cols(event_df, vertex_df):
    edge_out = unpack(vertex_df, 'out')
    edge_in = unpack(vertex_df, 'in')
    shower_df = event_df.join(edge_out)
    shower_df = shower_df.join(edge_in)
    edge_df = shower_df[['in', 'out']]
    max_id = edge_df.stack().max()
    isna = shower_df['out'].isna()
    final = shower_df['final']
    num_final = np.sum(final)
    final_ids = -1 * np.arange(max_id + 1, max_id + num_final + 1)
    if not shower_df[isna]['final'].all():
        raise RuntimeError(
            'Failed to add edges! Some outgoing vertices are not defined. '
            + 'Please report this to maintainers.'
            )
    shower_df.loc[(final, 'out')] = final_ids
    shower_df['out'] = shower_df['out'].astype('<i')
    return shower_df
