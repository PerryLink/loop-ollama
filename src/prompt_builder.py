"""SysPrompt builder-Ollama循环 S/A>=7B完整 B适中 C/D极简+few-shot 标准库"""
class PromptBuilder:
  _P={"init":("理解","拆解需求"),"analyzing":("规划","分析代码定计划"),
      "executing":("执行","读写文件运行命令"),"converging":("验证","检查结果确认完成")}
  def __init__(self,model_grade="A",tools=None):
    self.g=model_grade.upper()if model_grade else"A";self.t=tools or[]
  def _l(self):g=self.g;return"h"if g in("S","A")else"m"if g=="B"else"l"
  def _n(self,p):return self._P.get(p,self._P["init"])[0]
  def _d(self,p):return self._P.get(p,self._P["init"])[1]

  def build_system_prompt(self,phase="init",task="")->str:
    lv=self._l();tt=self.get_tool_definitions_text()
    n=self._n(phase);d=self._d(phase);t=task or"待指定"
    ts=f"## 可用工具\n{tt}\n"if tt else""
    if lv=="h":
      return(f"顶级编程助手。阶段：{n}。\n角色：精通编程、系统设计、代码审查。"
             f"分析定位问题。严守指令。\n任务：{t}\n指引：{d}\n{ts}"
             f"规范：1.一次一工具等结果 2.先搜后读先读后改 "
             f"3.Bash绝对路径描述用途 4.不确定先问\n"
             f"安全：禁删用户文件、禁破坏命令(rm -rf)、禁泄露密钥/Token\n"
             f"输出：先意图后调用，完成总结变更，出错给方案")
    if lv=="m":
      return(f"编程助手。阶段：{n}。\n角色：读代码改文件跑命令。\n"
             f"任务：{t}\n指引：{d}\n{ts}"
             f"规则：1.一次一工具等结果 2.先读后改 3.绝对路径 4.不确定问 5.不删不危")
    return(f"编程助手。阶段：{n}。任务：{t}\n指引：{d}\n\n"
           f"规则：1.一次一工具 2.先读后改 3.绝对路径 4.不确定问\n\n"
           f"示例：\n读: read file_path=\"/p\"\n"
           f"改: edit file_path=\"/p\" old_string=\"...\" new_string=\"...\"\n"
           f"跑: bash command=\"ls /p\" description=\"用途\"\n\n"
           f"{ts if ts else'(无工具，纯文本回复)'}")

  def build_turn_prompt(self,state,last_observation="")->str:
    t=state.get("task","");r=state.get("turn",1)
    x=state.get("max_turns",10);p=state.get("phase","init")
    m=state.get("modified_files",[]);c=state.get("converged",False)
    n=self._n(p)
    pts=[f"## 任务\n{t}",f"## 进度\n回合 {r}/{x} — {n}",
         f"## 收敛\n{'已收敛请确认'if c else'未收敛继续'}"]
    if m:pts.append("## 已修改\n"+"\n".join(f"- {f}"for f in m))
    if last_observation:pts.append(f"## 上轮\n{last_observation}")
    pts.append("## 行动\n选一个工具执行。先意图后调用。")
    return"\n\n".join(pts)

  def get_tool_definitions_text(self)->str:
    """Ollama工具定义格式化为文本块。"""
    if not self.t:return""
    ls=[]
    for i,x in enumerate(self.t):
      f=x.get("function",x)
      ls.append(f"- **{f.get('name',f'tool_{i}')}**: {f.get('description','')}")
      pp=f.get("parameters",{}).get("properties",{})
      rq=f.get("parameters",{}).get("required",[])
      for k,v in pp.items():
        r="*"if k in rq else"";d=v.get("description",str(v))if isinstance(v,dict)else str(v)
        ls.append(f"  {k}{r}: {d}")
    return"\n".join(ls)

  def build_convergence_prompt(self,state)->str:
    t=state.get("task","");n=self._n(state.get("phase","converging"))
    return(f"## 收敛检查\n任务：{t}\n阶段：{n}\n\n"
           f"1.要求满足？2.有无遗漏？3.输出符合预期？\n"
           f"全部完成回复**CONVERGED**并总结。未完成说明还需做什么。")

  def build_degraded_prompt(self,state)->str:
    t=state.get("task","");r=state.get("turn",1)
    m=state.get("modified_files",[]);ms=",".join(m)if m else"无"
    return(f"降级模式——无法调用工具。\n\n"
           f"任务：{t}\n回合：{r}\n已修改：{ms}\n\n"
           f"描述下一步操作及预期工具调用，给代码或命令示例。")
