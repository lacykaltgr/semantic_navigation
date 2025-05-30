import logging
from collections import defaultdict
import numpy as np
import pandas as pd

# Initialize logging
logging.basicConfig(level=logging.DEBUG, filename='/tmp/mapping_process.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
class MappingTracker:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MappingTracker, cls).__new__(cls)
            # Initialize the instance "once"
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if not self.__initialized:
            self.curr_frame_idx = 0
            self.curr_object_count = 0
            self.total_detections = 0
            self.total_objects = 0
            self.total_merges = 0
            self.merge_list = []
            self.object_dict = {}
            self.curr_class_count = defaultdict(int)
            self.total_object_count = 0
            self.prev_obj_names = []
            self.prev_bbox_names = []
            self.brand_new_counter = 0

            
    def increment_total_detections(self, count):
        self.total_detections += count
    def get_total_detections(self):
        return self.total_detections
    
    def set_total_detections(self, count):
        self.total_detections = count
    
    def increment_total_detections(self, count):
        self.total_detections += count
    
    def get_total_operations(self):
        return self.total_operations
    
    def set_total_operations(self, count):
        self.total_operations = count
    
    def increment_total_operations(self, count):
        self.total_operations += count
    
    def get_total_objects(self):
        return self.total_objects

    def set_total_objects(self, count):
        self.total_objects = count

    def increment_total_objects(self, count):
        self.total_objects += count
        
    def track_merge(self, obj1, obj2):
        self.total_merges += 1
        self.merge_list.append((obj1, obj2))
        
    def increment_total_merges(self, count):
        self.total_merges += count


class DenoisingTracker:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DenoisingTracker, cls).__new__(cls)
            # Initialize the instance "once"
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if not self.__initialized:
            self.total_operations = 0
            self.efficiency = defaultdict(int)
            self.object_stats = defaultdict(self._default_object_stats)
            # Initialize bucket_stats properly
            self.bucket_stats = defaultdict(self._default_bucket_stats)
            self.size_buckets = self._define_size_buckets()
            self.max_bucket = 0
            self.efficiency_keys = [
                ("No Change", 0),  # You might want to keep this as is or change to a preferred term
                ("<1%", 1),
                ("<5%", 5),
                ("<10%", 10),
                ("<30%", 30),
                ("<50%", 50),
                ("<70%", 70),
                ("<90%", 90),
                ("<100%", 100),
            ]
            self.__initialized = True

    @staticmethod
    def _define_size_buckets():
        return [(0, 50), (51, 100), (101, 200), (201, 500), (501, 1000),
                (1001, 2000), (2001, 3000), (3001, 5000), (5001, 10000),
                (10001, 100000), (100001, 1000000), (1000001, 10000000),
                (10000001, 100000000), (100000001, 1000000000), (1000000001, float('inf'))]
        
    @staticmethod
    def _default_object_stats():
        return {
            "denoise_count": 0,
            "no_point_removal_count": 0,
            "consecutive_no_removal_streak": 0,
            "max_consecutive_no_removal": 0,
            "original_size": 0,
        }

    @staticmethod
    def _default_bucket_stats():
        # This method returns a dictionary structured for storing denoising statistics
        # with updated, shorter key names for efficiency metrics.
        return {
            "denoise_count": 0,
            "No Change": 0,  # Assuming you're keeping "no_change" as "No Change" or adjust as necessary
            "<1%": 0,
            "<5%": 0,
            "<10%": 0,
            "<30%": 0,
            "<50%": 0,
            "<70%": 0,
            "<90%": 0,
            "<100%": 0,
            "points_removed": [],
            "percent_removed": [],
        }

    def get_size_bucket(self, size):
        for start, end in self.size_buckets:
            if start <= size <= end:
                return (start, end)
        return (1000001, float('inf'))

    def track_denoising(self, object_id, original_count, new_count):
        self.total_operations += 1
        reduction = original_count - new_count
        reduction_percentage = (reduction / original_count) * 100 if original_count else 0
        bucket = self.get_size_bucket(original_count)
        
        self.max_bucket = max(self.max_bucket, bucket[0])
        
        # if bucket[0] >= 1000001:
        #     # throw an error 
        #     raise ValueError(f"Object size {original_count} is too large for the defined size buckets")
        
        object_stat = self.object_stats[object_id]
        object_stat["denoise_count"] += 1
        object_stat["original_size"] = original_count
        
        bucket_stat = self.bucket_stats[bucket]
        bucket_stat["denoise_count"] += 1
        bucket_stat.setdefault("points_removed", []).append(reduction)
        bucket_stat.setdefault("percent_removed", []).append(reduction_percentage)
        
        if reduction == 0:
            bucket_stat["No Change"] += 1
            self.efficiency["No Change"] += 1
            object_stat["no_point_removal_count"] += 1
            object_stat["consecutive_no_removal_streak"] += 1
        else:
            object_stat["max_consecutive_no_removal"] = max(object_stat["max_consecutive_no_removal"], object_stat["consecutive_no_removal_streak"])
            object_stat["consecutive_no_removal_streak"] = 0
            for key, threshold in self.efficiency_keys:
                if reduction_percentage < threshold:
                    bucket_stat[key] += 1
                    self.efficiency[key] += 1
                    break

    def generate_report(self):
        data = []
        total_operations_across_buckets = 0
        all_points_removed = []  # Collect all points removed for overall average and median
        all_percent_removed = []  # Collect all percent removed for overall average and median
        totals_for_keys = {key: 0 for key, _ in self.efficiency_keys}  # Initialize totals for each key

        for bucket, stats in self.bucket_stats.items():
            total_operations_across_buckets += stats["denoise_count"]
            points_removed = stats.get("points_removed", [])
            percent_removed = stats.get("percent_removed", [])
            all_points_removed.extend(points_removed)  # Aggregate points removed
            all_percent_removed.extend(percent_removed)  # Aggregate percent removed

            avg_points_removed = np.mean(points_removed) if points_removed else 0
            median_points_removed = np.median(points_removed) if points_removed else 0
            avg_percent_removed = np.mean(percent_removed) if percent_removed else 0
            median_percent_removed = np.median(percent_removed) if percent_removed else 0

            row_data = {
                "Sort Key": bucket[0],  # Numeric sort key for sorting
                "Bucket": f"{bucket[0]}-{bucket[1]}",
                "Denoise Count": stats["denoise_count"],
                "Avg Points Removed": avg_points_removed,
                "Median Points Removed": median_points_removed,
                "Avg Percent Removed": avg_percent_removed,
                "Median Percent Removed": median_percent_removed,
            }

            # Update totals for each key
            for key, _ in self.efficiency_keys:
                category_count = stats.get(key, 0)
                totals_for_keys[key] += category_count  # Accumulate totals across all buckets
                if stats["denoise_count"] > 0:
                    percentage_of_total = (category_count / stats["denoise_count"]) * 100
                    row_data[key] = f"{percentage_of_total:.2f}%"
                else:
                    row_data[key] = "N/A"

            data.append(row_data)

        # Calculate overall averages and medians
        overall_avg_points_removed = np.mean(all_points_removed) if all_points_removed else 0
        overall_median_points_removed = np.median(all_points_removed) if all_points_removed else 0
        overall_avg_percent_removed = np.mean(all_percent_removed) if all_percent_removed else 0
        overall_median_percent_removed = np.median(all_percent_removed) if all_percent_removed else 0

        # Append the totals row with calculated overall statistics
        totals_row = {
            "Bucket": "Total",
            "Denoise Count": total_operations_across_buckets,
            "Avg Points Removed": f"{overall_avg_points_removed:.2f}",
            "Median Points Removed": f"{overall_median_points_removed:.2f}",
            "Avg Percent Removed": f"{overall_avg_percent_removed:.2f}",
            "Median Percent Removed": f"{overall_median_percent_removed:.2f}",
        }

        # Calculate and add percentages for the totals row
        for key, _ in self.efficiency_keys:
            if total_operations_across_buckets > 0:
                percentage_of_total = (totals_for_keys[key] / total_operations_across_buckets) * 100
                totals_row[key] = f"{percentage_of_total:.2f}%"
            else:
                totals_row[key] = "N/A"

        data.append(totals_row)  # Add the totals row to the data

        # Creating DataFrame
        df = pd.DataFrame(data)
        df = df.sort_values(by="Sort Key", ascending=True)  # Ensure the total row is at the bottom
        df = df.drop(columns=["Sort Key"], errors='ignore')  # Safely drop the 'Sort Key' column

        # Define column order, excluding "Sort Key"
        column_order = ["Bucket", "Denoise Count", "Avg Points Removed", "Median Points Removed",
                        "Avg Percent Removed", "Median Percent Removed"] + [key for key, _ in self.efficiency_keys]

        # Reorder the DataFrame based on 'column_order'
        df = df[column_order]

        print(df.to_string(index=False))  # Print the DataFrame without the index


        logging.info(f"\n{df.to_string()}")  # Log the DataFrame without the index
        print(f"Max bucket: {self.max_bucket}")
        k=1
