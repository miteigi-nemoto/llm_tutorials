#source code for article https://vc.ru/ml/1098324-sravnenie-summarizacii-v-mixtral-8x7b-instruct-pri-fp16-8-bit-4-bit-bonus-primery-iz-cloude-3

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import numpy as np
torch.manual_seed(0)
np.random.seed(0)
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
#MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
#cache_dir = '/miteigi_ext/hugging_cache/'


bnb_config4 = BitsAndBytesConfig(  
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= True,
    #llm_int8_enable_fp32_cpu_offload= True
)

#Конфиг для 8бит - подставьте в quantization_config=
bnb_config8 = BitsAndBytesConfig(  
    load_in_8bit= True, 
    llm_int8_threshold=200.0
    #llm_int8_enable_fp32_cpu_offload= True
)

torch.cuda.empty_cache()

import gc
gc.collect()

#запуск в 4бит
model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config4,
        device_map="auto",
         
        torch_dtype=torch.float16,
        #trust_remote_code=True,
        #cache_dir=cache_dir
        )


from transformers import pipeline, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=cache_dir)

import gc
gc.collect()

abstract =  """
   
 
   В   глубине  Западного  спирального  рукава  Галактики,  в  захолустье,
которого даже нет на картах, лежит небольшое желтое безвестное солнце.
     Вокруг  него по  орбите радиусом примерно  девяносто два миллиона  миль
обращается  крайне  незначительная  зелено-голубая  планетка.  Ее  жизненные
формы, произошедшие от обезьян,  так изумительно примитивны, что до  сих пор
считают электронные часы расчудесной выдумкой.
     У этой планеты есть -- или, точнее, была -- такая проблема: большинство
живших  на ней  людей  были несчастливыми  чуть  ли  не  все  время.  Для ее
разрешения было предложено  много  рецептов,  большинство которых  сводилось
преимущественно  к перемещению зеленых кусочков  бумаги, что странно, потому
что, вообще говоря, зеленые кусочки бумаги несчастными не были.
     Итак, задача  не решалась.  Многие  грустили,  а  большая  часть  людей
пребывала в отчаянии, даже обладатели электронных часов.
     Многие люди все  больше укреплялись во мнении,  что крупная ошибка была
сделана, прежде всего, тогда, когда все поспускались с деревьев. А некоторые
говорили,  будто  даже  залезание  на  деревья  было ошибкой,  и  никому  не
следовало покидать океаны.
     И  вот, однажды в  четверг, приблизительно через  две  тысячи лет после
того, как одного человека прибили к дереву за то,  что он рассказывал людям,
как  чудесно было бы для  разнообразия подобреть друг к другу, одна  девушка
посиживала  в маленьком кафе в Рикмэнсуорсе.  И внезапно она  поняла, что же
было не так все это время. Теперь она знала, как можно было бы сделать  свой
мир  добрым и  счастливым местом. На этот раз  все было верно, все могло  бы
получиться, и никого не нужно было бы ни к чему прибивать.
     К  сожалению,  прежде чем  она  сумела  добраться  до  телефона,  чтобы
кому-нибудь об  этом рассказать, разразилась  ужасная, слепая  катастрофа, а
идея была утеряна навсегда.
     Эта история не о ней.
     Это история ужасной, слепой катастрофы и некоторых из ее последствий.
     Еще  это  история  книги,  называющейся  "Путеводитель  "Автостопом  по
Млечному  Пути".  Неземной  книги, которую никогда не публиковали  на Земле.
Книги,  которую  до кошмарной катастрофы не видел, и  о которой не слышал ни
один землянин.
     И, несмотря на последнее, совершенно замечательной книги.
     Действительно,  она,  вероятно,  была  самой  примечательной  из  книг,
когда-либо выпущенных гигантской издательской корпорацией Малой Медведицы, о
которой ни один землянин также никогда не слыхал.
     Не  только совершенно замечательной была книга, но еще и очень удачной.
Она  была  популярнее, чем "Руководство по  божественному  уходу  за домом",
продавалась лучше, чем "Еще 53 способа проделать это при нулевой гравитации"
и оказалась еще более спорной, чем трилогия философских супербоевиков Оолона
Коллафида "В чем Бог был не прав", "Еще о величайших ошибках Бога" и "Да кто
он такой, этот Бог, в конце концов?"
     Во   многих  патриархальных  цивилизациях  Внешнего   восточного  обода
Галактики "Путеводитель "Автостопом по  Млечному Пути"  уже  отнял у великой
"Encyclopaedia  Galactica"  славу  стандартного  вместилища всего  знания  и
мудрости.   Хотя  в   Путеводителе   много  пробелов,  и   содержится  масса
недостоверного,  или,  по  меньшей мере, ужасно неточного, он  выигрывает по
сравнению со старой,  более прозаической,  энциклопедией в двух существенных
отношениях.
     Во-первых, он  слегка  дешевле, а  во-вторых,  на его обложке  большими
дружелюбными буквами написаны слова "НЕ ПУГАЙСЯ".
     А  история  того  ужасного, бессмысленного  четверга, его  чрезвычайных
последствий  и того, как нерасторжимо  они связаны с  замечательной  книгой,
начиналась очень просто.
     Она началась с дома.


        Глава 1

1


     Дом стоял на  пологом  склоне,  на самом краю  городка. Он стоял себе и
смотрел вдаль,  на широкие просторы фермерских полей Западной  Англии. Ничем
не примечательный  дом, построенный около тридцати  лет назад.  Приземистый,
угловатый, сложенный из  кирпича, дом нес на своем фасаде четыре окна, таких
размеров  и пропорций, которые,  в  большей  или меньшей  степени, не  могли
порадовать глаз.
     Единственным  человеком, для которого дом имел  какое-то значение,  был
Артур  Дент, да и то потому, что ему довелось там жить. Он  жил в доме около
трех лет,  все время после переезда из Лондона, вызванного наводимой большим
городом  нервозностью  и раздражительностью. Артуру, как  и дому, было около
тридцати  лет.  Высокий,  темноволосый,  он  никогда не  пребывал  в  полном
согласии с самим собой. Обычно его больше всего беспокоили  вопросы людей  о
том, почему он выглядит таким обеспокоенным. Артур работал на местном радио,
и  всегда  говорил друзьям, что  это  гораздо интереснее, чем они, наверное,
думают.  Друзья  именно так  и думали.  Кстати,  многие из  них  работали  в
рекламе.
     В среду ночью прошел очень сильный дождь, улица стала мокрой и грязной,
но утреннее  солнце четверга ярко и ясно светило на дом Артура Дента потому,
что делало это в последний раз.
     Артура еще  не известили в установленном порядке,  что  городской совет
хочет снести дом и проложить на его месте шоссе.

2


     Утром  четверга,  в  восемь  часов,  Артур  чувствовал  себя не  лучшим
образом. Он проснулся, как в дурмане. Встал, словно оглушенный,  побродил по
комнате, открыл  окно,  увидел бульдозер, нашел свои шлепанцы и  потащился в
ванную умываться.
     Пасту на щетку, -- так. Почистил.
     Зеркальце  для  бритья показывало в  потолок.  Он  его повернул. На миг
зеркальце отразило второй бульдозер в окне ванной. Повернутое правильно, оно
отражало  щетину.  Сбрил  щетину,  умылся, вытерся  и  прошлепал  на  кухню,
пожевать вкусненького.
     Чайник, крышка, холодильник, молоко, кофе. Зевок.
     Слово  бульдозер  поблуждало  миг  в  сознании,  в  поисках,  с чем  бы
соединиться. Бульдозер за кухонным окном был очень здоровым.
     Артур уставился на него.
     -- Желтый, -- подумал он и зашлепал назад, в спальню, чтобы одеться.
     Проходя мимо ванной, остановился выпить большой  стакан воды, потом еще
один.  Это  подозрительно  походило  на  похмелье.   Откуда  похмелье?   Пил
предыдущим  вечером?  Артур  предположил,  что,  должно  быть,  пил. Блик  в
бритвенном зеркальце. "Желтый", -- подумал Артур и потащился в спальню.
     Он стоял и соображал.  Пивная. О Господи, пивная.  Смутно припомнилось,
что  был  зол. Злился из-за  чего-то,  казавшегося важным. Говорил  об  этом
что-то,  и,  наверное,  очень  долго,  потому  что  самым  ясным  зрительным
воспоминанием  были  стеклянные взгляды на лицах слушателей. Что-то  о новой
дороге, --  это все, что удалось вспомнить. Ею занимались несколько месяцев,
только, кажется, об этом никто не  знал. Возмутительно. Артур  глотнул воды.
Все разрешится само собой, решил он. Никому дорога не  нужна,  у  городского
совета нет поддержки. Все образуется само по себе.
     Господи,  какое  жуткое,  однако,  похмелье. Глянул  на себя  в зеркало
платяного шкафа. Высунул язык. "Желтый", -- подумал он. Слово желтый бродило
в уме, подыскивая, к чему прилепиться.
     Пятнадцатью  секундами  позже  Артура  в доме  не  было: он лежал перед
большим желтым бульдозером, надвигавшимся на его садовую дорожку.

3


     Мистер Л. Проссер был, как говорится,  всего-навсего человеком. Другими
словами, он представлял собою углеродную двуногую  форму жизни, произошедшую
от обезьяны. Точнее говоря, сорокалетнего, жирного,  потрепанного  работника
городского совета.  Довольно любопытно, что он также являлся прямым потомком
Чингиз-хана  по мужской  линии,  хотя сам  этого  не  знал.  Правда,  череда
поколений и смешение рас так перетасовали гены, что явные монголоидные черты
отсутствовали,  и  единственными  рудиментами,  унаследованными мистером  Л.
Проссером от своего могучего предка, оставались пресловутые крепость желудка
и пристрастие к маленьким меховым шапкам.
     Мистер Проссер  ни  в  каком  смысле  не был  великим  воином,  -- лишь
нервным,  тревожным  человеком.  Сегодня  он  был  особенно   обеспокоен   и
встревожен, поскольку  всерьез  не ладилась работа, состоявшая в том,  чтобы
обеспечить снос дома Артура Дента до конца дня.
     -- Оставьте, мистер Дент. Вы ведь  знаете,  что  не сможете настоять на
своем. Вы  не можете лежать перед  бульдозером  бесконечно, -- сказал мистер
Проссер  и попытался заставить свои глаза яростно засверкать, но они  просто
не были на такое способны.
     Артур лежал в грязи и выражал свой протест.
     -- Поспорим и посмотрим, кто первым заржавеет!
     -- Боюсь, вам придется смириться, -- сказал мистер Проссер, хватаясь за
свою  меховую  шапку и  елозя  ею  по  макушке.  --  Это шоссе  должно  быть
построено, и оно будет построено!
     -- Впервые слышу! Почему оно должно быть проложено?
     Мистер  Проссер  немного  погрозил  Артуру пальцем, затем  остановился,
убрал палец и переспросил:
     --  Что  вы  имеете в виду под "Почему  оно должно быть построено?" Это
ведь шоссе. Вам нужно, чтобы строили дороги.
     Шоссе -- это приспособления, которые позволяют одним людям очень быстро
мчаться из  пункта  А в  пункт Б,  в то  время, как другие люди очень быстро
несутся из пункта  Б в  пункт  А.  Людям,  живущим в пункте В, расположенном
прямо посередине, остается удивляться, чем так хорош пункт А, что  множество
жителей пункта  Б страстно стремятся туда, и  чем так  восхитителен пункт Б,
что  так много жителей пункта А так сильно хотят туда попасть. Жители пункта
В часто  желают им  всем, раз и навсегда, попасть  ко всем  чертям, куда они
хотят.
     Мистеру  Проссеру  хотелось в  пункт  Г.  Пункт  Г  был  не  то,  чтобы
определенным  местом, -- просто любым удобным  местечком подальше от пунктов
А,  Б и В.  Ему  хотелось  бы  иметь  в  пункте  Г  миленький  коттеджик, со
скрещенными топорами над дверью, и проводить в пункте Д  (ближайшей к пункту
Г пивной),  столько  времени,  сколько  было бы  приятно.  Жена, разумеется,
предпочла бы вьющиеся розы, а он -- топоры. Неизвестно, почему, -- просто из
любви к топорам. Мистер Проссер горячо покраснел под насмешливыми  ухмылками
бульдозеристов.
     Он  переминался  с  ноги  на  ногу, но на любой из них чувствовал  себя
одинаково неуютно. Ясно, что кто-то был ужасающе не прав, и  он  молил Бога,
чтобы это оказался не он сам.
     -- Вы же знаете, что  имели полное право внести любые предложения или в
положенный срок заявить протест.
     --  В положенный срок? -- завопил Артур. --  Положенный срок? Я впервые
узнал  обо всем вчера вечером, когда появился рабочий. Я  спросил, не пришел
ли  он  мыть окна, а  он ответил,  что нет -- разрушить дом. Конечно,  он не
говорил  прямо.  О нет.  Сначала вытер  пару  окон  и содрал  с меня  за это
пятерку. Только потом сказал.
     --   Но   мистер  Дент,   планы   можно   было  свободно  посмотреть  у
проектировщиков в течение последних девяти месяцев.
     -- О да,  сразу после  услышанного я прямиком пошел их смотреть. Вчера,
после обеда.  Вы ведь  совсем не изменили своему обыкновению,  афишируя  их,
правда? Я имею в виду, рассказывая кому-нибудь о чем-нибудь.
     -- Но планы были вывешены на доске объявлений...
     --  Вывешены? На самом  деле мне  пришлось спуститься  в подвал,  чтобы
найти их.
     -- Там отдел информации.
     -- С фонарем.
     -- А, ну, лампочки, наверное, вышли из строя.
     -- Как и лестницы.
     -- Но, послушайте, вы ведь нашли объявление. Разве нет?
     --  Да,  --  сказал  Артур. --  Да, нашел.  Оно  было вывешено  на  дне
запертого на ключ шкафа, сваленного в неработающей уборной, а на дверях было
написано "Берегись леопарда".
     В небе проплывало облако. Оно бросило тень на Артура Дента,  лежавшего,
опираясь локтем  в холодную грязь. Бросило тень на дом  Артура Дента. Мистер
Проссер хмуро глянул на облако.
     -- Это не то, чтобы особенно симпатичный дом, -- сказал он.
     -- Извините, уж так получилось, что он мне нравится.
     -- Вам понравится шоссе.
     -- А, заткнитесь,  --  сказал Артур.  -- Заткнитесь, уберитесь отсюда и
возьмите свое чертово шоссе с собой. Вам ничего не сделать, и вы знаете это.
     Рот  мистера Проссера пару раз открылся и закрылся,  а его рассудок  на
мгновение  заполнили  невыразимые,  но  ужасно привлекательные видения:  дом
Артура Дента, пожираемый огнем,  и сам Артур, вопящий над пылающими руинами,
с,  по  меньшей мере, тремя здоровенными копьями,  прошедшими насквозь через
спину.  Мистера Проссера часто беспокоили похожие  видения, от чего он очень
нервничал. Он запнулся на минуту, и снова взял себя в руки.
     -- Мистер Дент!
     -- Привет? Да?
     --  Немного  фактов  для вас. Вы  имеете  представление  о  том,  какие
повреждения получит бульдозер, если я пущу его прокатиться прямо поверх вас?
     -- Какие? -- спросил Артур.
     --  Вовсе  никаких,  --  ответил мистер  Проссер,  и  отошел, испытывая
нервное потрясение, недоумевая, почему  мозг заполнен тысячей орущих на него
волосатых всадников.

"""

example = {'instruction' : 'Составь 10 самых важных событий из книги.',
    'input' : abstract }
def formatting_func(example):
    text = f"<s>[INST]Используя текст выше, ответь СТРОГО НА РУССКОМ ЯЗЫКЕ, иначе оштрафую тебя. Если не знаешь ответ, напиши: Пока не знаю ответ, задавай больше вопросов, чтобы я стала умнее.{example['instruction']}\n\n### Input:{example['input']}[/INST]"
    return text
eval_prompt = formatting_func(example)
#print(eval_prompt)

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

from datetime import datetime
import re

def model_seq_gen(model) : 
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
        start = datetime.now()
        sequences = pipe(
            f'{eval_prompt}' ,
            do_sample=True,
            
            max_new_tokens=1500, 
            temperature=0.01, 
            pad_token_id=tokenizer.eos_token_id
            #top_p=0.95
        )
        extracted_title = re.sub(r'[\'"]', '', sequences[0]['generated_text'].split("[/INST]")[1])
        stop = datetime.now()
        time_taken = stop-start
        print(f"Execution Time : {time_taken}")
        return extracted_title

extracted_title = model_seq_gen(model)
print(extracted_title)
