import pandas as pd

class data_generation():
    def __init__(self, dir_path):

        self.route_assignment_data = pd.read_csv(str(dir_path) + "route_assignment.csv")
        self.link_performance_data = pd.read_csv(str(dir_path) + "link_performance.csv")

        # origin data
        self.ozone_df = self.route_assignment_data.groupby('o_zone_id')['volume'].sum()
        self.ozone_df = self.ozone_df.reset_index()
        self.ozone_df['o_node_id'] = self.ozone_df['o_zone_id']

        # origin-destination data
        self.od_df = self.route_assignment_data.groupby(['o_zone_id', 'd_zone_id'])['volume', 'travel_time'].sum()
        self.od_df = self.od_df.reset_index()
        self.od_df['od_id'] = self.od_df.index + 1

        # path data
        self.path_df = self.route_assignment_data[['o_zone_id', 'd_zone_id', 'node_sequence']]
        self.path_df['path_id'] = self.path_df.index + 1

        # link data
        self.link_df = self.link_performance_data[['link_id', 'from_node_id', 'to_node_id', 'fftt', 'capacity']]
        self.link_df['link_no'] = self.link_df.index

    def origin_layer(self):
        # origin layer
        node_zone_dict = self.ozone_df[['o_node_id', 'o_zone_id']].set_index('o_node_id').to_dict()['o_zone_id']
        return node_zone_dict

    def origin_dest_layer(self):
        # origin-destination layer
        node_zone_dict = self.origin_layer()
        self.od_df['od_pair'] = self.od_df.apply(lambda x: (int(x.o_zone_id), int(x.d_zone_id)), axis=1)
        self.od_df['o_zone_id'] = self.od_df.apply(lambda x: node_zone_dict[int(x.o_zone_id)], axis=1)
        od_pair_dict = self.od_df[['od_pair', 'od_id']].set_index('od_pair').to_dict()['od_id']
        return od_pair_dict

    def od_to_path_layer(self):
        od_pair_dict = self.origin_dest_layer()
        self.path_df['od_id'] = self.path_df.apply(lambda x: od_pair_dict[int(x.o_zone_id), int(x.d_zone_id)], axis=1)
        # path_od_dict = self.path_df[['path_id', 'od_id']].set_index('path_id').to_dict()['od_id']
        return self.path_df

    def path_link_layer(self):
        self.link_df['link_pair'] = self.link_df.apply(lambda x: (int(x.from_node_id), int(x.to_node_id)), axis=1)
        link_no_pair_dict = self.link_df[['link_no', 'link_pair']].set_index('link_pair').to_dict()['link_no']
        return link_no_pair_dict

    def incidence_mat(self):
        # To count the number of x_f flow variables
        link_no_pair_dict = self.path_link_layer()
        path_df = self.od_to_path_layer()
        path_no1 = 0
        od_path1_idx_list = []
        path1_link_idx_list = []
        path2_link_idx_list = []
        for i in range(len(self.od_df)):
            # pdb.set_trace()
            od_id = self.od_df.loc[i, 'od_id']
            path_df_od = path_df[path_df['od_id'] == od_id].reset_index(drop=True)

            for j in range(len(path_df_od)):
                node_sequence = list(map(int, path_df_od.loc[j, 'node_sequence'].split(';')[0: -1]))
                link_sequence = [link_no_pair_dict[(node_sequence[k], node_sequence[k + 1])] for k in range(len(node_sequence) - 1)]

                if j < len(path_df_od) - 1:
                    od_path1_idx_list.append((i, path_no1))
                    for link_id in link_sequence:
                        path1_link_idx_list.append((path_no1, link_id))
                    path_no1 += 1
                else:
                    for link_id in link_sequence:
                        path2_link_idx_list.append((i, link_id))

        return od_path1_idx_list, path1_link_idx_list, path2_link_idx_list