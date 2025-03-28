"""
    This file contains data-generator for model
"""
from tensorflow.keras.utils import Sequence
import numpy as np

from spine_segmentation.segmentation.utils import open_dcm_prjs, open_png_prjs

from typing import Callable, Literal

class Data_train_generator(Sequence):
    """
        Training generator for Unet++ model of spine-segmentation problem

        Output `__get_item__`
        ---------------------
        list `[batch_x, {"output_1": batch_y, "output_2": batch_y, "output_3": batch_y, "output_4": batch_y}]`

        if mode == `side | frontal` then `batch_shape` = `(batch_size, height, width, 1)`\n
        if mode == `both` then `batch_shape` = `(batch_size, 2, height, width, 1)`\n
    """
    def __init__(self, x_files_list: dict, y_files_list: dict, batch_size: int, mode: Literal["side", "frontal", "both"] = "side", w_part: Literal["heap", "spine"] = "spine", deep_supervision: bool = True, update_func: Callable | None = None, new_image_size: tuple[int, int] = None, **kwargs) -> None:
        """
            Training generator constructor\n

            Parameters:
            -----------
                x_files_list:
                    dictionary like `{"side": [paths_to_dicom...], "frontal": [paths_to_dicom...]}`
                \n
                y_files_list:
                    dictionary like `{"side": [paths_to_png...], "frontal": [paths_to_png...]}`
                \n
                batch_size:
                    Size of batch
                \n
                mode:
                    `side | frontal | both` - witch data to be loaded - side projection, frontal, or both
                \n
                w_part:
                    `heap | spine` - witch data to be watched - heap or spine
                \n
                deep_supervision:
                    `true/false` - 4 outputs (for every layer), 1 output (the last layer)
                \n
                update_func:
                    Function which is called after every epoch
                \n
                new_image_size:
                    Reshape input to new shape  
        """
        super().__init__(**kwargs)
        self.data: dict[Literal["side", "frontal"], str] = x_files_list
        self.labels: dict[Literal["side", "frontal"], str] = y_files_list
        self.batch_size: int = batch_size
        
        self.deep_supervision: bool = deep_supervision
        
        self.new_image_size: tuple[int, int] | None = new_image_size
        self.mode: Literal["side", "frontal", "both"] = mode
        self.w_part: Literal["spine", "hip"] = w_part

        self.l_d: int = min(len(self.data["side"]), len(self.data["frontal"])) if mode == "both" else (len(self.data["side"]) if mode == "side" else len(self.data["frontal"]))

        self.update_func: Callable | None = update_func

    def __len__(self):
        return int(np.ceil((self.l_d) / float(self.batch_size)))

    def __getitem__(self, index):
        batch_x, batch_y = [], []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            if i < self.l_d:
                x = open_dcm_prjs(self.data["side"][i], self.data["frontal"][i], self.new_image_size)
                y = open_png_prjs(self.labels["side"][i], self.labels["frontal"][i], self.w_part, self.new_image_size)
                if self.mode == "both":
                    batch_x.append(x)
                    batch_y.append(y)
                else:
                    mode_ind = 0 if self.mode == "side" else 1
                    batch_x.append(x[mode_ind])
                    batch_y.append(y[mode_ind])
        
        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = np.array(batch_y, dtype=np.float32)

        return (batch_x, {"output_1": batch_y, "output_2": batch_y, "output_3": batch_y, "output_4": batch_y}) if self.deep_supervision else (batch_x, {"output_4": batch_y})

    def on_epoch_end(self):
        if self.update_func != None:
            self.update_func()

class CrossValidation_train_generator:
    """
        CrossValidation class for Data_train_generator
    """
    def __init__(self, x_files_list: dict, y_files_list: dict, batch_size: int, val_size: int, mode: Literal["side", "frontal", "both"] = "side", w_part: Literal["heap", "spine"] = "spine", deep_supervision: bool = True, new_image_size = None) -> None:
        """
            Training generator constructor\n

            Parameters:
            -----------
                x_files_list:
                    dictionary like `{"side": [paths_to_dicom...], "frontal": [paths_to_dicom...]}`
                \n
                y_files_list:
                    dictionary like `{"side": [paths_to_png...], "frontal": [paths_to_png...]}`
                \n
                batch_size:
                    Size of batch
                \n
                val_size:
                    Count of validation images
                \n
                mode:
                    `side | frontal | both` - witch data to be loaded - side projection, frontal, or both
                \n
                w_part:
                    `heap | spine` - witch data to be watched - heap or spine
                \n
                deep_supervision:
                    `true/false` - 4 outputs (for every layer), 1 output (the last layer)
                \n
                new_image_size:
                    Reshape input to new shape  
        """
        self.data: dict[Literal["side", "frontal"], str] = x_files_list
        self.labels: dict[Literal["side", "frontal"], str] = y_files_list

        self.batch_size: int = batch_size
        self.mode: Literal["side", "frontal", "both"] = mode
        self.w_part: Literal["spine", "hip"] = w_part

        self.deep_supervision = deep_supervision
        self.new_image_size: tuple[int, int] | None = new_image_size

        self.val_size: int = val_size
        self.l_d: int = min(len(self.data["side"]), len(self.data["frontal"]))

        self.update_count = 0
        x, y = self._get_data()

        self.data_gen = Data_train_generator(x[0], y[0], self.batch_size, self.mode, self.w_part, self.deep_supervision, self._update_gen, self.new_image_size)
        self.val_gen = Data_train_generator(x[1], y[1], self.batch_size, self.mode, self.w_part, self.deep_supervision, self._update_gen, self.new_image_size)

    def _get_data(self) -> tuple[
        list[dict[Literal["side", "frontal"], str], dict[Literal["side", "frontal"], str]], 
        list[dict[Literal["side", "frontal"], str], dict[Literal["side", "frontal"], str]]
    ]:
        """
            Returns tuple of `files dicts of dicoms` and `files dicts of pngs`\n
            `([train_dcm, val_dcm], [train_png, val_png])`
        """
        return ([
            {
                "side": self.data["side"][:-self.val_size],
                "frontal": self.data["frontal"][:-self.val_size]
            },
            {
                "side": self.data["side"][-self.val_size:],
                "frontal": self.data["frontal"][-self.val_size:]
            }
        ],
        [
            {
                "side": self.labels["side"][:-self.val_size],
                "frontal": self.labels["frontal"][:-self.val_size]
            },
            {
                "side": self.labels["side"][-self.val_size:],
                "frontal": self.labels["frontal"][-self.val_size:]
            }
        ])
    
    def _update_gen(self):
        """
            Actual shuffling
        """
        if self.update_count < 2:
            self.update_count += 1
        else:
            self.update_count = 0

            sh_indexes = np.random.shuffle(
                np.arange(start=0, stop=self.l_d, step=1, dtype=np.int32)
            )
            for s in ["side", "frontal"]:
                self.data[s] = [self.data[s][i] for i in sh_indexes]
                self.labels[s] = [self.labels[s][i] for i in sh_indexes]
            
            x, y = self._get_data()

            self.data_gen = Data_train_generator(x[0], y[0], self.batch_size, self.mode, self.w_part, self.deep_supervision, self._update_gen, self.new_image_size)
            self.val_gen = Data_train_generator(x[1], y[1], self.batch_size, self.mode, self.w_part, self.deep_supervision, self._update_gen, self.new_image_size)
