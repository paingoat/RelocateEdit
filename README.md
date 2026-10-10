# RelocateEdit

Di chuyển một vật trong ảnh: khoanh một vòng quanh vật, bấm điểm muốn đặt nó tới, rồi xem pipeline chạy từng bước.

Ứng dụng là giao diện Gradio. Mỗi giai đoạn có nút riêng, xếp từ trên xuống dưới, để nhìn được mask nguồn, depth, mask đích, nền đã xóa và ảnh cuối.

Repo này chỉ có code. Weight được tải trên máy chạy (RunPod), không tải và không chạy thử trên máy đang viết code.

## Pipeline

```text
ảnh + vòng khoanh                         điểm đích (mũi tên)
        |                                        |
        v                                        |
   1. SEEM  -> mask vật nguồn                    |
        |                                        |
ảnh -> 2. Depth Anything V2 -> depth map        |
        |                                        |
        +------------------+---------------------+
                           v
              3. Dời mask, scale theo depth
                           |
              4. LaMa xóa vật ở vị trí cũ  -> nền sạch
                           |
              5. AnyDoor chèn vật vào mask đích -> ảnh cuối
```

| Bước | Model | Việc nó làm | Vì sao chọn |
| --- | --- | --- | --- |
| 1 | SEEM Focal-L v0 | Từ một prompt hình học ra mask của đúng một vật | Demo của SEEM nhận stroke và trả về một mask. Không có prompt bbox thật. |
| 2 | Depth Anything V2 Large | Depth tương đối của cả ảnh | Gọn, không cần compile. Bản metric (mét) là tùy chọn. |
| 3 | code trong `ops/relocation.py` | Tịnh tiến + scale mask | Không phải model. Scale lấy từ depth tại chỗ vật chạm nền. |
| 4 | LaMa big-lama + feature refinement | Xóa vật nguồn, giữ nền | Generator FFC chạy được mà không cần cả stack training. Refinement là phần xử lý ảnh lớn của paper Feature Refinement. |
| 5 | AnyDoor | Vẽ lại vật vào vùng đích | Nhận ảnh vật + mask, và mask vùng cần chèn. Bám silhouette khi bật shape control. |

## Vòng khoanh trở thành prompt của SEEM

SEEM được học với nét vẽ **nằm trên vật**. Vòng người dùng vẽ **bao quanh** vật thì những pixel đó là nền. Đưa thẳng vòng đó vào model sẽ hỏi "vật nào trông giống vành nền này".

Bước 1 làm như sau (`relocate_edit/ops/scribble.py`):

1. Khép các chỗ đứt nhỏ, tô kín vùng bên trong vòng, lấy bbox của vùng đó.
2. Erode vùng này rồi lấy điểm xa biên nhất (đỉnh của distance transform). Đó là cách SEEM tự biến một box thành một chấm khi đánh giá.
3. Nở chấm đó khoảng 3 pixel, thêm vài điểm bên trong nữa.
4. Chạy một lượt SEEM. Thay vì chỉ lấy query giống prompt nhất, xếp hạng lại 101 mask:

```text
score = 0.5 * softmax(độ giống prompt) + 0.3 * (phần mask nằm trong vòng) + 0.2 * IoU(bbox mask, bbox vòng)
```

5. Giữ mask điểm cao nhất, lấy thành phần liên thông lớn nhất, đưa về đúng cỡ ảnh gốc.

Nếu muốn tô thẳng lên vật như demo gốc của SEEM, chọn chế độ "Tô trực tiếp lên vật".

## Scale theo depth và biên ảnh

Tâm mask là điểm neo. Điểm người dùng bấm là chỗ tâm đó phải tới.

Depth tham chiếu là median ở dải dưới cùng của mask (chỗ vật chạm nền), không phải median cả vật. Công thức lặp 3 lần vì điểm chạm ở đích còn phụ thuộc chính scale:

- Depth tương đối (số lớn = gần hơn): `scale = d_đích / d_nguồn`
- Depth metric, đơn vị mét (số lớn = xa hơn): `scale = Z_nguồn / Z_đích`

Sau đó nhân với hệ số trên UI và kẹp vào `[s_min, s_max]` (mặc định 0.2 đến 5).

Phép biến đổi là scale đều quanh tâm rồi tịnh tiến. Không xoay.

Khi vật sau khi dời bị tràn khỏi ảnh:

| Chế độ | Hành vi |
| --- | --- |
| `shift` (mặc định) | Dịch cả vật vào trong ảnh, giữ nguyên scale. |
| `shrink` | Giảm scale cho đến khi vật vừa khung, vẫn hướng về điểm đã bấm. |
| `clip` | Giữ nguyên vị trí, cắt phần tràn. Mask vật đưa vào AnyDoor cũng bị cắt phần tương ứng, để model không phải vẽ nốt phần đã nằm ngoài ảnh. |

Bước 3 còn vẽ một ảnh "dán thử": cắt pixel gốc rồi đặt sang chỗ mới. Đây không phải ảnh sinh. Ảnh sinh là bước 5.

## Máy cần gì

Chạy trên Linux có GPU NVIDIA. Script cài `nvcc` 12.8 bằng apt (repo CUDA có sẵn trên image RunPod) và wheel PyTorch `cu128`. GPU Blackwell (RTX PRO 4500, dòng RTX 50, kiến trúc `sm_120`) không có kernel trong wheel CUDA 11.8. Không cài `cuda-toolkit` bằng conda: solver báo `InvalidSpec` với meta-package đó trên `linux-64`.

| Hạng mục | Mức |
| --- | --- |
| GPU | 24 GB trở lên. AnyDoor để fp32 khoảng 11 GB weight, cộng SEEM và depth khoảng 1.3 GB mỗi cái. Trên 24 GB hãy bật `--offload` để mỗi lúc chỉ một model nằm trên GPU. 40 GB có thể để cả bốn model trên GPU. |
| RAM | Khoảng 32 GB khi giải nén checkpoint AnyDoor. File gốc khoảng 17 GB vì còn optimizer state. Script chỉ giữ `state_dict`. |
| Đĩa | Khoảng 60 GB trong lúc tải. Cache Hugging Face nằm ở `/workspace/data` (tạo sẵn nếu chưa có). Sau khi xóa file gốc AnyDoor còn khoảng 30 GB. |
| Hệ điều hành | Ubuntu trên RunPod. Script dùng `apt-get`, Miniconda và `bash`. |

Depth Anything V2 **Large** có weight CC-BY-NC-4.0. Bản Small (Apache-2.0) không phải mặc định vì bản Large ổn định hơn cho ảnh thường.

## Hugging Face

Không repo nào trong pipeline bắt buộc bấm approve trên Hugging Face. API `gated` của từng repo là `false`:

| Thành phần | Repo | Approve |
| --- | --- | --- |
| SEEM Focal-L | `xdecoder/SEEM` | Không |
| Depth Anything V2 Large | `depth-anything/Depth-Anything-V2-Large` | Không. License weight là CC-BY-NC-4.0, nhưng repo không khóa tải |
| Depth metric (tùy chọn) | `depth-anything/Depth-Anything-V2-Metric-Hypersim-Large`, `...-VKITTI-Large` | Không |
| big-lama | `smartywu/big-lama` | Không |
| AnyDoor | space `xichenhku/AnyDoor` | Không |
| Tokenizer CLIP của SEEM | `openai/clip-vit-base-patch32` | Không |

Vẫn nên đặt token. File AnyDoor khoảng 17 GB, tải ẩn danh dễ bị giới hạn tốc độ. Token loại Read là đủ: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## Cài đặt

Trên pod, clone repo rồi điền token trước khi tải weight:

```bash
cd RelocateEdit
cp .env.example .env
# Sửa HF_TOKEN trong .env
bash scripts/setup.sh
```

`.env` không được commit. `HF_HUB_ENABLE_HF_TRANSFER=1` trong file mẫu bật `hf_transfer` (đã cài trong env) để tải song song, nhanh hơn tải tuần tự.

Cache Hugging Face nằm ở `/workspace/data/huggingface`. Script tạo `/workspace/data` nếu chưa có. Checkpoint dùng lúc chạy nằm ở `weights/`, là symlink tới `/workspace/data/weights` trên RunPod, nên cả hai còn sau khi restart pod.

`setup.sh` chạy lần lượt:

1. `scripts/00_install_miniconda.sh` — cài Miniconda vào `/workspace/miniconda3` (còn sau khi restart pod). Nếu không có `/workspace` thì cài vào `$HOME/miniconda3`.
2. `scripts/01_create_env.sh` — env conda tên `relocate`, Python 3.10, `nvcc` 12.8 từ apt, PyTorch 2.7.1+cu128, xformers 0.0.31, detectron2 (build từ fork `MaureenZOU/detectron2-xyz`, có `sm_120`), rồi `requirements.txt` (gồm `hf_transfer`).
3. `scripts/02_download_weights.sh` — đọc `.env`, tải weight vào `/workspace/data`.

Tải thêm depth metric (trong nhà / ngoài trời) và weight DINOv2 rời:

```bash
bash scripts/setup.sh --with-metric --with-dino
```

DINOv2 rời không bắt buộc. Checkpoint AnyDoor đã chứa weight ViT-g. Chỉ cần file rời nếu muốn nạp encoder trước khi nạp AnyDoor (`anydoor.dino_weight` trong yaml).

Cài lại riêng từng phần: `bash scripts/01_create_env.sh` hoặc `bash scripts/02_download_weights.sh`.

Nếu app báo `no kernel image is available for execution on the device`, env vẫn đang dùng PyTorch CUDA 11.8. Chạy lại `bash scripts/01_create_env.sh`. Weight đã tải thì không cần chạy lại `setup.sh`.

Env dùng một bộ thư viện cho cả bốn model. Code training và các pin xung đột nhau (Lightning 1.2 với 1.5, Gradio 3 với 4, transformers 4.19) không được cài. Transformers ở đây là 4.36.2 để vừa gọi được CLIP tokenizer của SEEM vừa thỏa Gradio. Chi tiết phần code giữ lại và chỗ đã sửa nằm ở [third_party/README.md](third_party/README.md).

Pillow bị khóa ở 9.5 vì detectron2 còn gọi `Image.LINEAR`, hàm đã bị xóa ở Pillow 10. Sau khi build detectron2, script cài lại `setuptools==69.5.1`: bản 82 trở lên bỏ `pkg_resources`. App cũng tự tạo `pkg_resources.declare_namespace` nếu module đó vắng, vì checkpoint LaMa và AnyDoor đều import Lightning. Script cài GCC 11 và dùng nó khi build detectron2, để Ubuntu 22.04 và 24.04 cùng một compiler. CUDA 12.8 chấp nhận GCC 11.

## Chạy Gradio, có link public

```bash
bash scripts/run_app.sh
```

Script bật `share=True`. Terminal in một URL dạng `https://....gradio.live`. Mở URL đó từ máy khác. Pod phải ra được internet để Gradio tạo tunnel.

Trên GPU 24 GB:

```bash
bash scripts/run_app.sh --offload
```

Nạp sẵn cả bốn model lúc mở app (chậm hơn, nhưng bước đầu không phải chờ):

```bash
bash scripts/run_app.sh --offload --preload
```

Cổng mặc định `7860`, lắng nghe `0.0.0.0`. Đổi cổng: `bash scripts/run_app.sh --server-port 7861`.

Mỗi bước ghi ảnh và `info.json` vào `output/<YYYY-MM-DD_HH-mm-ss>/`, trong các thư mục `A`, `B`, `1`, `2`, `3`, `4`, `5`.

Test không cần weight, không cần GPU (cần numpy và scipy):

```bash
python -m unittest discover -s tests -v
```

## Cách dùng giao diện

Làm từ trên xuống.

**Bước A.** Kéo ảnh vào khung bên trái. Chọn cọ đỏ và vẽ một vòng kín quanh vật. Vòng không cần đẹp, nhưng nên khép. Bấm **Xác nhận nét khoanh**. Ảnh kèm nét đỏ hiện ở khung bên dưới.

**Bước B.** Bấm một điểm trên ảnh đó. Đó là chỗ tâm vật sẽ tới. App vẽ mũi tên từ tâm vòng khoanh tới điểm vừa bấm. Bấm lại để đổi điểm.

**Chạy toàn bộ pipeline** đi hết năm bước với các thông số đang hiện trên form.

Hoặc chạy từng nút:

1. **SEEM.** Chọn "Vòng khoanh quanh vật" hoặc "Tô trực tiếp lên vật". Kết quả: ảnh phủ mask, bbox, các chấm prompt, và ba mask điểm cao nhất. Json có tên lớp COCO, điểm số, bbox.
2. **Depth.** Mặc định là disparity tương đối, màu ấm là gần. Hai chế độ metric cần weight tải bằng `--with-metric`. Ảnh này chỉ để xem; số thật nằm trong mảng depth dùng ở bước 3.
3. **Dời mask.** Chỉnh hệ số scale và cách xử lý khi tràn ảnh. Kết quả: mask đích màu xanh, ảnh dán thử, scale, depth nguồn, depth đích, tỷ lệ còn nằm trong ảnh.
4. **LaMa.** Nở mask (mặc định 15 pixel) để gỡ viền và bóng. Bật feature refinement khi ảnh lớn hơn khoảng 1024 pixel cạnh ngắn; với ảnh nhỏ refiner gần như chỉ forward một lần. Kết quả: mask đã nở và nền đã xóa vật.
5. **AnyDoor.** Cần đủ ba bước SEEM, dời mask và LaMa. Shape control mặc định bật vì mask đích là silhouette đã scale, không phải hình chữ nhật. Guidance cao thì vật giống ảnh gốc hơn và hòa nền kém hơn. Seed được gieo cho cả numpy và torch. Kết quả: ảnh cắt vật trên nền trắng, ảnh cuối, và ảnh trước/sau.

Đổi nét khoanh sẽ xóa mọi kết quả. Đổi điểm đích sẽ xóa bước dời mask và bước chèn. Chạy lại một bước cũng xóa các bước phụ thuộc nó.

## Cấu hình

Sửa [`configs/default.yaml`](configs/default.yaml). Đường dẫn tương đối tính từ gốc repo.

Những chỗ hay đụng:

- `device`, `offload`
- `seem.prompt_mode`, `seem.dilation`, ba trọng số `w_sim`, `w_in`, `w_iou`
- `depth.mode` và đường dẫn checkpoint
- `relocate.s_min`, `relocate.s_max`, `relocate.boundary`
- `lama.dilation`, `lama.refine`, `lama.px_budget`
- `anydoor.steps`, `anydoor.guidance`, `anydoor.strength`, `anydoor.seed`, `anydoor.shape_control`, `anydoor.save_memory`

`anydoor.save_memory: true` bật sliced attention và chuyển VAE / DINOv2 ra khỏi GPU trong lúc denoise. Chậm hơn, đỡ tốn VRAM.

## Thêm hoặc đổi một model

Mỗi model là một class trong `relocate_edit/models/`, kế thừa `BaseModelWrapper`: `load()` đọc weight, một method public chạy một ảnh. `relocate_edit/registry.py` gọi đúng class theo tên. Pipeline trong `relocate_edit/pipeline.py` không import code của SEEM, LaMa hay AnyDoor trực tiếp.

Muốn đổi bộ segment, viết wrapper mới trả về mask boolean cùng cỡ ảnh, rồi trỏ registry vào class đó. Bước dời mask, xóa và chèn không cần sửa.

Code gốc nằm trong `third_party/`. Đừng sửa upstream để thêm tính năng của app. Tính năng mới để trong `relocate_edit/`.

## License

- Code SEEM: MIT
- Code Depth Anything V2: Apache-2.0. Weight Large / Giant / bản metric: CC-BY-NC-4.0
- Code LaMa: Apache-2.0
- Code AnyDoor và DINOv2: Apache-2.0

Mỗi thư mục trong `third_party/` giữ file LICENSE của upstream.
