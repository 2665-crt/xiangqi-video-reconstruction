import base64
import json
import os
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from time import monotonic

import cv2
import httpx
import numpy as np
from pydantic import ValidationError

from .chess import apply_move, legal_moves
from .schemas import Piece, Position, Square


class VisionRequestError(ValueError):
    def __init__(self, message, code, phase, retryable=True, outcome_uncertain=False, retry_after_s=None):
        super().__init__(message)
        self.code, self.phase = code, phase
        self.retryable, self.outcome_uncertain = retryable, outcome_uncertain
        self.retry_after_s = retry_after_s


def retry_after_seconds(value):
    if not value:
        return None
    try:
        return max(0, float(value))
    except ValueError:
        try:
            return max(0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return None

PROMPT = """你是中国象棋棋盘识别器。输入是相邻视频截图，红方在下，可能有手部遮挡或短暂走棋变化。
逐张分析完整局面。相同局面用多帧消除模糊；不同局面分别给完整候选，不把不同时刻的棋子拼成一个局面。
识别棋盘的9列10行交叉点：file从左到右0..8，rank从上到下0..9。图像外缘可能有留白。
color只允许英文red或black；kind只允许英文king、advisor、elephant、horse、rook、cannon、pawn。
对应将帅、士仕、象相、马、车、炮、兵卒，不使用中文字或其他字段名。
结合所有截图消除文字模糊和手部遮挡，不默认标准开局。不能看清时给候选，不补造棋子。
返回结构为JSON对象，唯一顶层字段candidates是数组。每项有pieces数组和side_to_move。
pieces中每项有color、kind、file、rank；file和rank是整数。每个候选包含所有在场棋子。
字段格式示例（仅展示字段，不代表输入棋局）：{"candidates":[{"pieces":[{"color":"red","kind":"king","file":4,"rank":9},{"color":"black","kind":"king","file":4,"rank":0}],"side_to_move":null}]}。
同一交叉点最多一枚棋子，每方将帅1枚。输入可能是特殊摆局，其他棋子按实际数量识别，不以标准开局数量限制或补齐。
红帅只能在file=3..5、rank=7..9；黑将只能在file=3..5、rank=0..2。不要输出违反这些限制的候选。
静态截图通常不能知道下一方，side_to_move请用null；若明确可判断才写red或black。
无法判断棋盘时返回{"candidates":[]}。每个候选是完整局面，需要包含红帅和黑将。
不要生成棋谱、评价或置信度，不要输出Markdown。"""

ACTUAL_PROMPT = """你是实际象棋局面复刻器。按时间顺序观察截图，结合提供的已知局面判断操作结果。
只返回最后一张图中操作完成后的完整局面候选；前面的图片只用于辨认被拿起棋子的身份和原位置。
手遮挡或拿在手中属于操作过程，应结合前后图追踪。最后仍未落下或不足以确定完整局面，返回空candidates。
不要把前后不同时间的局面混在一起，不要把短暂拿起当成永久移除；不能因为手曾遮挡就拒绝清晰的最终局面。
包括正常走棋、吃子、悔棋、摆棋、放回原位、多枚调整、增减棋子。忠实按画面判断，象棋轮次与合法走法不是限制。
不默认标准开局，不补造看不见的棋子。特殊摆局可缺将帅、将帅在九宫外，甚至空棋盘。
截图已校正红方朝下。file从左到右0..8，rank从上到下0..9，是交叉点而不是格子。
返回JSON对象，唯一顶层字段candidates为完整最终局面的数组；每项pieces与side_to_move。
pieces每项仅有color(red或black)、kind(king/advisor/elephant/horse/rook/cannon/pawn)、file、rank。
同一交叉点最多一枚。side_to_move可以是null；无需用猜测下一方来阻止局面识别。
只有最终局面确实存在多个解释才给多个候选，无法确定时给空数组。不要输出Markdown。"""

CHANGE_PROMPT = """按时间顺序比较操作前后棋盘，仅核对最后画面中已完成的变化。
拿起、手部停顿、遮挡不是完成的操作。保留没有变化的棋子身份，红车不会因移动变成黑车或兵。
提供的已知局面是操作前的记忆。可以走棋、吃子、悔棋、摆棋、增减及同色换子，不强制合法走法。
只返回JSON：{"changes":[{"file":0,"rank":9,"color":null,"kind":null},{"file":0,"rank":8,"color":"red","kind":"rook"}]}。
示例仅展示格式。变化数组必须同时包含清空的起点与落点的最终棋子，不返回未变化的交叉点。
file从左到右0..8，rank从上到下0..9。color为red/black，kind为king/advisor/elephant/horse/rook/cannon/pawn。
空位的color和kind都为null；仍未完成或无法确定时返回{"candidates":[]}；没有变化时返回{"changes":[]}。"""

def piece_sheet(image):
    from .analysis import read_board_cells
    cells = read_board_cells(image, strict=True)
    occupied = {s: color for s, color in cells.items() if color is not None}
    small = cv2.resize(image, (450, 500))
    sheet = np.full((max(1, (len(occupied) + 5) // 6) * 120, 720, 3), 245, np.uint8)
    for i, ((file, rank), color) in enumerate(occupied.items()):
        x, y = 25 + file * 50, 25 + rank * 50
        tile = cv2.resize(small[y-24:y+25, x-24:x+25], (94, 94))
        row, col = divmod(i, 6)
        sheet[row*120:row*120+94, col*120+13:col*120+107] = tile
        cv2.putText(sheet, f'{file},{rank} {color}', (col*120+5, row*120+112),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (20, 20, 20), 1)
    return sheet, occupied


def parse_positions(content: str, reference: Position | None = None, actual: bool = False) -> list[Position]:
    if not isinstance(content, str):
        raise ValueError("最终内容必须是包含candidates的JSON文本")
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    payload = json.loads(content)
    if actual and reference is not None and isinstance(payload, dict) and isinstance(payload.get('changes'), list):
        changes = payload['changes']
        squares = [Square(file=p['file'], rank=p['rank']) for p in changes]
        if len({(s.file, s.rank) for s in squares}) != len(squares):
            raise ValueError('变化中包含重复交叉点')
        pieces = [{'color': p.color, 'kind': p.kind, 'file': p.square.file, 'rank': p.square.rank}
                  for p in reference.pieces if p.square not in squares]
        for change in changes:
            if change.get('color') is None and change.get('kind') is None:
                continue
            pieces.append(change)
        payload = {'candidates': [{'pieces': pieces, 'side_to_move': None}]}
    if isinstance(payload, list):
        payload = {"candidates": [{"pieces": payload, "side_to_move": None}]}
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        raise ValueError("JSON中缺少candidates数组")
    positions = []
    for candidate in payload["candidates"][:4]:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("pieces"), list):
            raise ValueError("candidates中的每项必须包含pieces数组")
        if any(isinstance(p, dict) and {'color', 'file', 'rank'} <= p.keys() and 'kind' not in p
               for p in candidate['pieces']):
            raise ValueError('模型只返回了占位信息，缺少棋种kind，无法区分车、炮、兵等棋子；请换清晰画面重新识别或手动设置初始棋局')
        if any(not isinstance(p, dict) or not {"color", "kind", "file", "rank"} <= p.keys() for p in candidate["pieces"]):
            raise ValueError("pieces中的每项必须包含color、kind、file、rank")
        turn = candidate.get("side_to_move")
        if turn not in ("red", "black", None):
            raise ValueError("模型返回了未知走棋方")
        try:
            pieces = [Piece(id=f"{p['color']}_{p['kind']}_{p['file']}_{p['rank']}", color=p["color"],
                            kind=p["kind"], square=Square(file=p["file"], rank=p["rank"])) for p in candidate["pieces"]]
            if actual:
                observed = Position(pieces=pieces, side_to_move=turn or (reference.side_to_move if reference else 'red'))
                if reference is not None:
                    from .actual import _cells_position, reconcile_position
                    cells = {(x, y): None for x in range(9) for y in range(10)}
                    cells.update({(p.square.file, p.square.rank): p.color for p in pieces})
                    if sorted((p.color, p.kind, p.square.file, p.square.rank) for p in pieces) == sorted(
                            (p.color, p.kind, p.square.file, p.square.rank) for p in reference.pieces):
                        observed = reference.model_copy(deep=True)
                    else:
                        unchanged_types = all(next((p.kind for p in pieces if p.square == previous.square and p.color == previous.color), previous.kind)
                                              == previous.kind for previous in reference.pieces)
                        restored = _cells_position(reference, cells, [reference]) if unchanged_types else None
                        observed = restored or reconcile_position(reference, observed, [reference])
                positions.append(observed)
                continue
            if reference is not None:
                occupied = tuple(sorted((p.color, p.square.file, p.square.rank) for p in pieces))
                reachable = [reference] + [apply_move(reference, move) for move in legal_moves(reference)]
                matches = [p for p in reachable if tuple(sorted((piece.color, piece.square.file, piece.square.rank)
                           for piece in p.pieces)) == occupied]
                if len(matches) == 1:
                    # 仅在完整颜色和坐标唯一匹配已确认局面或合法一步时恢复身份，不补缺失棋子。
                    positions.append(matches[0])
                    continue
            if any(sum(p.color == color and p.kind == 'king' for p in pieces) != 1 for color in ('red', 'black')):
                continue
            if any(p.kind == 'king' and not (3 <= p.square.file <= 5 and
                   (7 <= p.square.rank <= 9 if p.color == 'red' else 0 <= p.square.rank <= 2)) for p in pieces):
                continue
            valid = [Position(pieces=pieces, side_to_move=color) for color in ([turn] if turn else ["red", "black"])]
        except ValidationError:
            continue
        positions.extend(valid)
    return positions


class VisionClient:
    def __init__(self, base_url=None, key=None, model=None, client=None):
        self.base_url = (base_url or os.getenv("VISION_API_BASE", "")).rstrip("/")
        self.key = key or os.getenv("VISION_API_KEY", "")
        self.model = model or os.getenv("VISION_MODEL", "")
        self.client = client or httpx.Client(timeout=httpx.Timeout(180, connect=15, write=30, pool=15), trust_env=False)
        self.usage = []
        self.last_response = None
        self.last_request_count = 1
        self.progress = None
        from .piece_classifier import local_piece_classifier
        self.local_classifier = local_piece_classifier() if model is None and client is None else None
        if os.getenv('VISION_FUSION') == '1' and model is None and client is None:
            from .fusion import local_fusion_classifier
            self.local_classifier = local_fusion_classifier()

    @property
    def configured(self):
        return bool(self.local_classifier or (self.base_url and self.key and self.model))

    def recognize_position(self, images: list[np.ndarray], reference: Position | None = None, actual: bool = False) -> list[Position]:
        if self.local_classifier is not None and (reference is None or getattr(self.local_classifier, 'supports_reference', False)):
            self.last_request_count = self.last_image_count = 0
            options = {'actual': actual}
            if reference is not None:
                options['reference'] = reference
            result = self.local_classifier.recognize_position(images, **options)
            self.last_response = self.local_classifier.last_response
            return result
        self.last_request_count = 1
        if not (self.base_url and self.key and self.model):
            raise VisionRequestError('请先配置视觉模型', 'NotConfigured', 'configuration', retryable=False)
        prompt = ACTUAL_PROMPT if actual else PROMPT
        request_images = list(images[-3:])
        if actual and reference is not None:
            prompt = CHANGE_PROMPT
            prompt += '\n已知局面及棋子身份（移动前的记忆，不代替最终画面）：' + reference.model_dump_json(by_alias=True)
        elif reference is None and min(images[-1].shape[:2]) >= 450 and np.ptp(cv2.cvtColor(images[-1], cv2.COLOR_BGR2GRAY)) > 30:
            sheet, occupied = piece_sheet(images[-1])
            from .analysis import read_board_cells
            if len(read_board_cells(images[-1], strict=True)) == 90 and occupied:
                request_images = request_images[-2:] + [sheet]
                prompt += '\n附加图为同一初始局面的棋子局部图，标签是file,rank与本地颜色检测。仍需从图像读取每枚棋子的kind。'
                prompt += '\n占位证据（仅供核对，不是输出格式）：\n' + '\n'.join(
                    f'{f},{r}: {color}' for (f, r), color in occupied.items())
                prompt += '\n请逐一辨认这些交叉点的棋种，不漏子，不按文字朝向推断颜色。'
        if reference is None:
            prompt += '\n必须从图像辨认每枚棋子的kind，不能只复述占位证据。输出必须是包含candidates的对象，不是裸数组。'
            prompt += '\n每枚必须同时包含color、kind、file、rank；kind不得省略或写null。不能确定棋种时返回{"candidates":[]}，不猜测填充。'
            prompt += '\n最终输出示例（仅说明格式，不代表当前局面）：\n'
            prompt += '{"candidates":[{"pieces":[{"color":"red","kind":"king","file":4,"rank":9},{"color":"black","kind":"king","file":4,"rank":0}],"side_to_move":null}]}'
        content = [{"type": "text", "text": prompt}]
        self.last_image_count = len(request_images)
        for image in request_images:
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok:
                raise ValueError("棋盘截图编码失败")
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(encoded).decode()}})
        body = {"model": self.model, "messages": [{"role": "user", "content": content}], "temperature": 0}
        if self.model == "qwen3.8-omni-flash":
            body.update(reasoning_effort="none", response_format={"type": "json_object"}, stream=True,
                        stream_options={"include_usage": True})
        started = monotonic()
        self.last_response = None
        diagnostic = {'response_started': False, 'received_chars': 0, 'trace': [], 'connection_phase': 'connect'}
        def trace(event, info):
            if 'connect_tcp' in event:
                diagnostic['connection_phase'] = 'tcp'
            elif 'start_tls' in event:
                diagnostic['connection_phase'] = 'tls'
            elif 'send_request' in event:
                diagnostic['connection_phase'] = 'upload'
            elif 'receive_response' in event:
                diagnostic['connection_phase'] = 'response'
            item = {'event': event, 'elapsed_s': round(monotonic() - started, 3)}
            if info.get('exception') is not None:
                item['exception_type'] = type(info['exception']).__name__
            diagnostic['trace'].append(item)
            diagnostic['trace'] = diagnostic['trace'][-30:]
            if self.progress:
                self.progress(diagnostic['connection_phase'])
        try:
            if body.get('stream'):
                final, finish, complete = [], None, False
                with self.client.stream('POST', self.base_url + '/chat/completions',
                                        headers={'Authorization': 'Bearer ' + self.key}, json=body,
                                        extensions={'trace': trace}) as response:
                    diagnostic.update(response_started=True, request_id=response.headers.get('x-request-id'))
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if monotonic() - started > 300:
                            raise httpx.ReadTimeout('total response deadline', request=response.request)
                        if not line.startswith('data:'):
                            continue
                        data = line[5:].strip()
                        if data == '[DONE]':
                            complete = True
                            break
                        chunk = json.loads(data)
                        if chunk.get('error'):
                            self.last_response = dict(diagnostic, request_error='StreamError')
                            raise VisionRequestError('视觉服务流式响应返回错误，诊断已保存；未自动重试',
                                                     'StreamError', 'response', outcome_uncertain=True)
                        if chunk.get('usage'):
                            self.usage.append(chunk['usage'])
                        if chunk.get('choices'):
                            choice = chunk['choices'][0]
                            text = choice.get('delta', {}).get('content')
                            if text:
                                final.append(text)
                                diagnostic['received_chars'] += len(text)
                                diagnostic.setdefault('first_content_s', round(monotonic() - started, 2))
                            finish = choice.get('finish_reason') or finish
                self.last_response = dict(diagnostic, content=''.join(final), finish_reason=finish, stream_complete=complete)
                if not complete and finish is None:
                    self.last_response['request_error'] = 'IncompleteStream'
                    raise VisionRequestError('视觉服务流式响应中断，未接收到完成标记；诊断已保存，未自动重试',
                                             'IncompleteStream', 'response', outcome_uncertain=True)
                if finish not in (None, 'stop'):
                    self.last_response['request_error'] = 'IncompleteResponse'
                    raise VisionRequestError('视觉服务未正常完成生成，结果未采用；诊断已保存',
                                             'IncompleteResponse', 'response', outcome_uncertain=True)
            else:
                response = self.client.post(self.base_url + "/chat/completions", headers={"Authorization": "Bearer " + self.key},
                                            json=body, extensions={'trace': trace})
                diagnostic.update(response_started=True, request_id=response.headers.get('x-request-id'))
                response.raise_for_status()
                payload = response.json()
                self.usage.append(payload.get("usage", {}))
                try:
                    choice = payload["choices"][0]
                    self.last_response = {"content": choice["message"]["content"], "finish_reason": choice.get("finish_reason")}
                except (KeyError, TypeError, IndexError):
                    raise ValueError("视觉服务响应缺少最终文本，请检查接口兼容性") from None
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            self.last_response = dict(diagnostic, http_status=status, elapsed_s=round(monotonic() - started, 2))
            raise VisionRequestError(f"视觉服务请求失败（HTTP {status}），请检查后端模型配置", f'HTTP{status}',
                                     'response', retryable=status in (408, 429, 500, 502, 503, 504),
                                     outcome_uncertain=status in (408, 500, 502, 504),
                                     retry_after_s=retry_after_seconds(exc.response.headers.get('retry-after'))) from None
        except httpx.RequestError as exc:
            elapsed = round(monotonic() - started, 2)
            self.last_response = dict(diagnostic, request_error=type(exc).__name__, elapsed_s=elapsed)
            phase = 'response'
            if isinstance(exc, httpx.ConnectTimeout):
                message = '视觉服务建立连接超时，请检查网络连接'
                phase = 'connect'
            elif isinstance(exc, httpx.ReadTimeout):
                message = '等待模型响应超时，服务可能繁忙或生成较慢'
            elif isinstance(exc, httpx.WriteTimeout):
                message = '发送棋盘截图超时，请检查网络上传连接'
                phase = 'upload'
            elif isinstance(exc, (httpx.ReadError, httpx.RemoteProtocolError)):
                message = '视觉服务或中间网关在响应期间断开连接'
            elif isinstance(exc, httpx.ConnectError):
                message = '视觉服务连接失败，请检查网络连接'
                phase = 'connect'
            elif isinstance(exc, httpx.PoolTimeout):
                message = '等待视觉服务连接资源超时'
                phase = 'pool'
            else:
                message = '视觉服务连接失败，请检查网络连接'
            raise VisionRequestError(f'{message}（已等待 {elapsed:.0f} 秒）；诊断已保存，未自动重试',
                                     type(exc).__name__, phase, outcome_uncertain=phase not in ('connect', 'pool')) from None
        except (json.JSONDecodeError, KeyError, TypeError, IndexError):
            self.last_response = dict(diagnostic, request_error='InvalidResponse', elapsed_s=round(monotonic() - started, 2))
            raise VisionRequestError('视觉服务响应格式不兼容或不完整，诊断已保存；未自动重试',
                                     'InvalidResponse', 'response', outcome_uncertain=True) from None
        try:
            result = parse_positions(self.last_response["content"], reference=reference, actual=actual)
        except json.JSONDecodeError:
            self.last_response["parse_error"] = "invalid_json"
            raise ValueError("视觉服务未返回完整有效的JSON，响应可能被截断；诊断已保存") from None
        except (ValueError, IndexError) as exc:
            self.last_response["parse_error"] = str(exc)
            raise ValueError(f"视觉服务棋局结构不匹配：{exc}；诊断已保存") from None
        return result

    def close(self):
        self.client.close()
