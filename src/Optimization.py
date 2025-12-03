import numpy as np
from pyeda.inter import *
from typing import Tuple, List, Optional
from itertools import product 

def max_reachable_marking(
    place_ids: List[str], 
    bdd: BinaryDecisionDiagram, 
    c: np.ndarray
) -> Tuple[Optional[List[int]], Optional[float]]:
    
    max_value = -float('inf') 
    best_marking = None
    
    # Lấy tất cả các đánh dấu thỏa mãn BDD
    satisfying_assignments = list(bdd.satisfy_all())

    if not satisfying_assignments:
        return None, None
    
    # Ánh xạ tên biến sang index trong vector c (và marking)
    name_to_index = {name: i for i, name in enumerate(place_ids)}
    
    for assignment in satisfying_assignments:
        
        current_marking = [0] * len(place_ids)
        current_value = 0.0

        # Lưu lại các biến đã được gán giá trị trong assignment
        assigned_indices = set()

        # 1. Gán giá trị cho các biến CÓ trong assignment (Ràng buộc BDD)
        for bdd_var, value in assignment.items():
            var_name = bdd_var.name
            
            try:
                idx = name_to_index[var_name]
                p_i = int(value)
                current_marking[idx] = p_i
                current_value += c[idx] * p_i
                assigned_indices.add(idx)
            except KeyError:
                continue

        # 2. Gán giá trị tối ưu cho các biến KHÔNG có trong assignment (Don't Care)
        # Trường hợp BDD = 1 (Test 006) sẽ rơi vào đây vì assignment là {}
        # PyEDA đôi khi tối ưu hóa assignment, nên ta cần quét lại tất cả indices.
        
        for idx in range(len(place_ids)):
            if idx not in assigned_indices:
                # Đây là biến don't care. Áp dụng quy tắc tối ưu:
                # Nếu c[idx] > 0, chọn p_i = 1 để tối đa hóa
                # Nếu c[idx] <= 0, chọn p_i = 0 để loại bỏ giá trị âm/không thêm 0
                
                p_i_opt = 1 if c[idx] > 0 else 0
                
                current_marking[idx] = p_i_opt
                current_value += c[idx] * p_i_opt
        
        # 3. Cập nhật giá trị tối đa
        if current_value > max_value:
            max_value = current_value
            best_marking = current_marking

    if best_marking is None:
        return None, None
    else:
        # Giá trị tối đa có thể là số nguyên, nhưng trả về float theo định nghĩa hàm
        return [int(m) for m in best_marking], float(max_value)