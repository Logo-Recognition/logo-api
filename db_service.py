from typing import Generator, List,Optional
from flask import jsonify, request
from psycopg2.extensions import connection, cursor
import psycopg2
from minio import Minio
from minio.commonconfig import CopySource
from contextlib import contextmanager



class Annote :
    def __init__(self, id, image_id, class_index, x1 ,y1, x2, y2 ):
        self.id = id
        self.image_id = image_id
        self.class_index = class_index
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

class Image :
    def __init__(self,image_id,width,height,image_name) :
        self.image_id = image_id
        self.width = width
        self.height = height
        self.image_name = image_name

@contextmanager
def get_cursor(conn:connection) -> Generator[cursor, None, None] :
    cur: cursor = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()

class DB_service :
    @staticmethod
    def create_class(conn: connection, classname: str) -> bool:
        if conn is None:
            print("error : Database connection is not available")
            return False
        if not classname:
            print("error : Classname is required")
            return False
        try:
            with get_cursor(conn) as cursor :
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(class_index) FROM Class")
                max_class_number = cursor.fetchone()[0]
                if max_class_number is None :
                    new_class_number = 0
                else:
                    new_class_number = max_class_number + 1
                cursor.execute("INSERT INTO Class (class_index, classname) VALUES (%s, %s)", (new_class_number, classname))
                print(f"created class_index: {new_class_number}, class name: {classname}")
                return True
        except (Exception, psycopg2.Error) as error:
            print("Error inserting new class", error)
            return False
    
    @staticmethod
    def delete_class(conn: connection, classname: str, minio: Minio) -> bool:
        if conn is None:
            print("error : Database connection is not available")
            return False
        if not classname:
            print("error : Classname is required")
            return False
        try:
            with get_cursor(conn) as cursor:
                annotes = DB_service.get_labels_by_classname(conn,classname)
                for annote in annotes :
                    if len(DB_service.get_labels_by_image_id(conn, annote.image_id)) <= 1:
                        cursor.execute("SELECT imagename FROM Image WHERE iid = %s;", (annote.image_id,))
                        image_name = cursor.fetchone()[0]
                        
                        cursor.execute("DELETE FROM Annote WHERE id = %s;", (annote.id,))
                        cursor.execute("DELETE FROM Image WHERE iid = %s;", (annote.image_id,))
                        
                        minio.copy_object('unannotated', image_name, CopySource('annotated', image_name))
                        minio.remove_object('annotated', image_name)
                        
                        print(f'Deleted image and moved: {image_name}')
                    else:
                        cursor.execute("DELETE FROM Annote WHERE id = %s;", (annote.id,))
                        print(f'Deleted annote id: {annote.id}')
                cursor.execute("SELECT class_index FROM Class WHERE classname = %s", (classname,))
                print(f'Deleted class : {classname} ')
                result = cursor.fetchone()
                if result is None:
                    print("class_index not found from classname")
                    return False
                class_index = result[0]
                cursor.execute("DELETE FROM Class WHERE classname = %s", (classname,))
                cursor.execute("UPDATE Class SET class_index = class_index - 1 WHERE class_index > %s", (class_index,))
                cursor.execute("UPDATE Annote SET class_index = class_index - 1 WHERE class_index > %s", (class_index,))
                print(f"deleted class_index: {class_index}, class name: {classname}")
                return True
        except (Exception, psycopg2.Error) as error:
            print("Error deleting class", error)
            return False
    
    @staticmethod
    def get_classes_name(conn: connection) -> list[str]:
        if conn is None:
            print("error : Database connection is not available")
            return []
        try:
            with get_cursor(conn) as cursor:
                cursor.execute("SELECT classname,class_index FROM Class ORDER BY class_index;")
                result = cursor.fetchall()
                classes = []
                if result is None:
                    print("Classes not found")
                    return []
                for row in result:
                    classes.append(row[0])
                print(f"classes : {classes}")
                return classes
        except (Exception, psycopg2.Error) as error:
            print("Error getting classes", error)
            return []

    @staticmethod
    def get_class_index_by_classname(conn: connection, classname: str) -> int:
        if conn is None:
            print("error : Database connection is not available")
            return None
        try:
            with get_cursor(conn) as cursor:
                cursor.execute("SELECT class_index FROM Class WHERE classname = %s", (classname,))
                result = cursor.fetchone()
                if result is None:
                    print("class_index not found")
                    return None
                class_index = result[0]
                print(f"get class_index: {class_index} from class name: {classname}")
                return class_index
        except (Exception, psycopg2.Error) as error:
            print("Error getting class_index ", error)
            return None
    
    @staticmethod
    def get_labels_by_classname(conn: connection, classname: str) -> Optional[list[Annote]]:
        if conn is None:
            print("error : Database connection is not available")
            return None
        try:
            labels = []
            with get_cursor(conn) as cursor:
                class_index = DB_service.get_class_index_by_classname(conn,classname)
                cursor.execute("SELECT * FROM Annote WHERE class_index = %s", (class_index,))
                result = cursor.fetchall()
                if result is None:
                    print("Labels not found")
                    return None
                for row in result :
                    labels.append(Annote(row[0],row[1],row[2],row[3],row[4],row[5],row[6]))
                return labels
        except (Exception, psycopg2.Error) as error:
            print("Error getting annotes", error)
            return None
    
    @staticmethod
    def get_labels_by_image_id(conn: connection, iid: int) -> Optional[list[Annote]]:
        if conn is None:
            print("error : Database connection is not available")
            return None
        try:
            labels = []
            with get_cursor(conn) as cursor:
                cursor.execute("SELECT * FROM Annote WHERE iid = %s", (iid,))
                result = cursor.fetchall()
                if result is None:
                    print("Label not found by iid")
                    return None
                for row in result :
                    labels.append(Annote(row[0],row[1],row[2],row[3],row[4],row[5],row[6]))
                return labels
        except (Exception, psycopg2.Error) as error:
            print("Error getting annotes ", error)
            return None

    @staticmethod
    def get_id_by_imagename(conn: connection, imagename:str) -> int:
        if conn is None:
            print("error : Database connection is not available")
            return None
        try:
            with get_cursor(conn) as cursor:
                cursor.execute("SELECT iid FROM Image WHERE imagename = %s", (imagename,))
                result = cursor.fetchone()
                if result is None:
                    print("iid not found from imagename")
                    return None
                return result[0]
        except (Exception, psycopg2.Error) as error:
            print("Error getting image ", error)
            return None

    @staticmethod
    def get_image_by_id(conn: connection, iid:int) -> Optional[Image]:
        if conn is None:
            print("error : Database connection is not available")
            return None
        try:
            with get_cursor(conn) as cursor:
                cursor.execute("SELECT * FROM Image WHERE iid = %s", (iid,))
                result = cursor.fetchone()
                if result is None:
                    print("Images not found by iid")
                    return None
                return Image(result[0],result[1],result[2],result[3])
        except (Exception, psycopg2.Error) as error:
            print("Error getting image ", error)
            return None

    @staticmethod
    def delete_image_by_id(conn: connection, iid:int) -> bool:
        if conn is None:
            print("error : Database connection is not available")
            return False
        try:
            with get_cursor(conn) as cursor:
                cursor.execute("DELETE FROM Image WHERE iid = %s;", (iid,))
                print(f"Deleted image id : {iid}")
                return True
        except (Exception, psycopg2.Error) as error:
            print("Error deleting image", error)
            return False

    @staticmethod
    def is_image_exist(conn :connection, imagename : str) -> bool :
        if conn is None:
            print("error : Database connection is not available")
            return False
        try:
            with get_cursor(conn) as cursor:
                cursor.execute("SELECT FROM Image WHERE imagename = %s;", (imagename,))
                result = cursor.fetchone()
                if result is None:
                    print("Images not found")
                    return False
                return True
        except (Exception, psycopg2.Error) as error:
            print("Error to find image", error)
            return False
        
    @staticmethod
    def create_annotes_by_iid(conn: connection, image_id:int , label_list):
        if conn is None:
            print("error : Database connection is not available")
            return False
        try:
            with get_cursor(conn) as cursor:
                for label in label_list :
                    class_index = DB_service.get_class_index_by_classname(conn,label['class_name'])
                    x1,y1,x2,y2 = label['bbox'].split()
                    cursor.execute("INSERT INTO Annote (iid, class_index, x1, y1, x2, y2) VALUES (%s, %s, %s, %s, %s, %s)",(image_id, class_index, x1, y1, x2, y2))  
                    print(f"create annote iid : {image_id} class_index: {class_index} bbox:{x1} {y1} {x2} {y2}")
                return True
        except (Exception, psycopg2.Error) as error:
            print("Error creating annotated image", error)
            return False

    @staticmethod
    def create_annotated_image(conn: connection,  image_name , width ,height, label_list) -> bool:
        if conn is None:
            print("error : Database connection is not available")
            return False
        try:
            with get_cursor(conn) as cursor:
                cursor.execute("INSERT INTO Image (width, height, imagename) VALUES (%s, %s, %s)", (width, height, image_name))
                print(f"create image name : {image_name} width : {width} height : {height}")
                cursor.execute("SELECT currval(pg_catalog.pg_get_serial_sequence('Image', 'iid'))")
                iid = cursor.fetchone()[0]
                if not DB_service.create_annotes_by_iid(conn,iid,label_list):
                    print("Error creating annotated", error)
                    return False
                return True
        except (Exception, psycopg2.Error) as error:
            print("Error creating annotated image", error)
            return False
    
    @staticmethod
    def get_annotated_images(conn: connection) -> list[dict]:
        if conn is None:
            print("error : Database connection is not available")
            return []
        get_annote_by_iid = """SELECT Annote.iid,Annote.class_index,Class.classname,Annote.x1,Annote.y1,Annote.x2,Annote.y2 FROM Annote JOIN Image ON Annote.iid = Image.iid
                            JOIN Class ON Annote.class_index = Class.class_index WHERE Image.iid = %s;"""
        annotated_images = []
        try:
            with get_cursor(conn) as cursor:
                cursor.execute("SELECT iid,imagename FROM Image")
                images = cursor.fetchall()
                if images is None:
                    print("Not found any images")
                    return []
                for image in images:
                    labels = []
                    cursor.execute(get_annote_by_iid,(image[0],))
                    annotes = cursor.fetchall()
                    for annote in annotes :
                        labels.append({
                            'class_id' : annote[1],
                            'class_name' : annote[2],
                            'bbox' : f'{annote[3]} {annote[4]} {annote[5]} {annote[6]}'
                        })
                    annotated_images.append({
                        'image' : image[1],
                        'label' : labels
                    })
                return annotated_images
        except (Exception, psycopg2.Error) as error:
            print("Error getting annotated images", error)
            return []

    @staticmethod
    def delete_all_annote_by_imagename(conn: connection,imagename) -> bool :
        if conn is None:
            print("error : Database connection is not available")
            return False
        try:
            with get_cursor(conn) as cursor:
                cursor.execute("SELECT iid FROM Image WHERE imagename = %s", (imagename,))
                image_id = cursor.fetchone()[0]
                if image_id is None :
                    print("error : Not found image")
                    return False
                cursor.execute("DELETE FROM Annote WHERE iid = %s;", (image_id,))
                return True
        except (Exception, psycopg2.Error) as error:
            print("Error deleting image", error)
            return False

    @staticmethod
    def get_annote_by_imagename(conn: connection,imagename) -> list[dict]:
        annote_list = []
        if conn is None:
            print("error : Database connection is not available")
            return False
        
        if not imagename:
            print("error : imagename is required")
            return False
        try:
            with get_cursor(conn) as cursor:
                cursor.execute("SELECT iid FROM Image WHERE imagename = %s", (imagename,))
                image_id = cursor.fetchone()
                if image_id:
                    image_id = image_id[0]
    
                    cursor.execute("SELECT x1,y1,x2,y2,class_index FROM Annote WHERE iid = %s", (image_id,))
                    annote_data  = cursor.fetchall()

                    if annote_data :
                        for value in annote_data:
                            annote_dict = {
                                'x1': value[0],
                                'y1': value[1],
                                'x2': value[2],
                                'y2': value[3],
                                'cid': value[4]
                            }
                            annote_list.append(annote_dict)
                        return annote_list 
                    else:
                        return None
                else:
                    return None
        except Exception as e:
            print("An error occurred:", e)
            return None
        finally:
            cursor.close()

               
