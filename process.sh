python /app/mapping_tools/clean_omni_data.py /workspace/data/data19/data19b/left
python /app/mapping_tools/clean_omni_data.py /workspace/data/data19/data19b2/left
python /app/mapping_tools/clean_omni_data.py /workspace/data/data19/data19c/left
python /app/mapping_tools/clean_omni_data.py /workspace/data/data19/data19d/left
python /app/mapping_tools/clean_omni_data.py /workspace/data/data19/data19e/left
python /app/mapping_tools/clean_omni_data.py /workspace/data/data19/data19f/left


./dense_recon /workspace/data/data19/data19a/left /workspace/data_proc/data19/data19a/dense19a_pcd.pcd
./dense_recon /workspace/data/data19/data19b/left /workspace/data_proc/data19/data19b/dense19b_pcd.pcd
./dense_recon /workspace/data/data19/data19b2/left /workspace/data_proc/data19/data19b2/dense19b2_pcd.pcd
./dense_recon /workspace/data/data19/data19c/left /workspace/data_proc/data19/data19c/dense19c_pcd.pcd
./dense_recon /workspace/data/data19/data19d/left /workspace/data_proc/data19/data19d/dense19d_pcd.pcd
./dense_recon /workspace/data/data19/data19e/left /workspace/data_proc/data19/data19e/dense19e_pcd.pcd

./map_merger /workspace/data_proc/data19/dense_merged.pcd \
                /workspace/data_proc/data19/data19a/dense19a_pcd.pcd \
                /workspace/data_proc/data19/data19b/dense19b_pcd.pcd \
                /workspace/data_proc/data19/data19c/dense19c_pcd.pcd \
                /workspace/data_proc/data19/data19d/dense19d_pcd.pcd \
                /workspace/data_proc/data19/data19e/dense19e_pcd.pcd \
                /workspace/data_proc/data19/data19b2/dense19b2_pcd.pcd \
                /workspace/data_proc/data19/data19f/dense19f_pcd.pcd

./filter /workspace/data_proc/data19/dense_merged_ds05.pcd /workspace/data_proc/data19/dense_merged_ds05_filtered.pcd
# ./filter /workspace/data_proc/data19/dense_merged.pcd /workspace/data_proc/data19/dense_merged_filtered.pcd
./downsample /workspace/data_proc/data19/dense_merged.pcd /workspace/data_proc/data19/dense_merged_ds05.pcd 0.05
# ./downsample /workspace/data_proc/data19/dense_merged_filtered.pcd /workspace/data_proc/data19/dense_merged_filtered_ds05.pcd 0.05
./z_thresh /workspace/data_proc/data19/dense_merged_ds05_filtered.pcd 2 /workspace/data_proc/data19/dense_merged_ds05_filtered_zt2.pcd
#  ./z_thresh /workspace/data_proc/data19/dense_merged_filtered_ds05.pcd 2 /workspace/data_proc/data19/dense_merged_filtered_ds05_zt2.pcd

cd /app/conceptfusion-compact/cf_compact/
python semantic_map.py 'dataset=data19/data19a'
python semantic_map.py 'dataset=data19/data19b'
python semantic_map.py 'dataset=data19/data19b2'
python semantic_map.py 'dataset=data19/data19c'
python semantic_map.py 'dataset=data19/data19d'
python semantic_map.py 'dataset=data19/data19e'

cd /app/mapping_tools
python merge_objects.py \
    /workspace/data_proc/data19/data19a/exps/mapping/pcd_mapping.pkl.gz \
    /workspace/data_proc/data19/data19b/exps/mapping/pcd_mapping.pkl.gz \
    /workspace/data_proc/data19/data19b2/exps/mapping/pcd_mapping.pkl.gz \
    /workspace/data_proc/data19/data19c/exps/mapping/pcd_mapping.pkl.gz \
    /workspace/data_proc/data19/data19d/exps/mapping/pcd_mapping.pkl.gz \
    /workspace/data_proc/data19/data19e/exps/mapping/pcd_mapping.pkl.gz \
    /workspace/data_proc/data19
