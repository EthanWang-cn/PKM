"""
 字符串格式化————python的字符串无法通过 + 和数字类型进行拼接 ，需要通过占位拼接
         %s(内容转字符串) %d（内容转整数） %f（内容转浮点数）
         精度控制  %m.nf  例如 %5.2f 设置精度为2,数字宽度为5
         快速格式化无法限制精度
         print(f"名字是{name},年龄{age}")
         print(name+ "%5.2f"%age)

var1 = input("该语句得到的永远是字符串类型，需要按需进行数据类型转换")

# 列表推导式
    even = [x for x in nums if x % 2 == 0]

# 元组组包和解包的用法
def swap():
    x, y, z = 10, 20, 30
    x, y, z = y, z, x
    print(x, y, z)  # 20 30 10

py容器类型
    list 列表：有序可变序列
    set 集合：无序去重集合
        #创建空元组
            tup1 = ()
    dict 字典：键值映射容器
        #创建空字典
            emptyDict = {}
            emptyDict = dict()

py函数
    传参
        位置传参
        关键字传参数
    形参可以有默认值，放在最后
    匿名函数: 通常作为高阶函数的参数
        lambda 参数列表:函数体
        lambda x, y: x + y
        shuchu = lambda : print("")
        data_list.sort(key=lambda item : len(item),reverse=True）

类型注解：Python中的一种语法特性，用于明确标识变量、函数参数和返回值的数据类型，从而使代码更清晰、更安全、更易维护。


模块导入
    from 包名 import 模块名
    from 模块名 import *
    import 包名.模块名.函数名
    #这个下面的不会被执行(正常导入时会运行导入的模块)
    if __name__ == "__main__":
    #__all__ 指定*指代哪些功能
    __all__=["函数名","","",""]

Package
    文件夹包含 __init__.py 描述当前包信息

面向对象

class 类名:
       def __init__(self, xxx,xxx):

异常
    try:
    except:
    finally:
    异常的传递


"""


# 元组组包和解包的用法
def swap():
    x, y, z = 10, 20, 30
    x, y, z = y, z, x
    print(x, y, z)  # 20 30 10
    nums = [x, y, z]
    # 列表推导式
    even = [x for x in nums if x % 2 == 0]
    print(even)
    a = dict()


if __name__ == '__main__':
    swap()
