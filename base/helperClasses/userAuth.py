from base.serializers import StockSerializer, UserModelSerializer, ActiveUserSerializer
from base.models import Stock, UserModel, ActiveUser
from django.utils import timezone
import random


def update_user_last_active(user_email):
    active_user = ActiveUser.objects.get(user_email=user_email)
    active_user.last_active = timezone.now()
    active_user.save()
    return


class UserAuth:
    def __int__(self):
        pass

    @staticmethod
    def check_if_user_active(user_email, access_token):
        ActiveUser.delete_inactive_users()
        if ActiveUser.objects.filter(user_email=user_email).exists():
            update_user_last_active(user_email)
            return ActiveUser.objects.get(user_email=user_email).access_token == access_token
        else:
            return False

    @staticmethod
    def check_creds(request):
        users = UserModel.objects.all()
        print(users, type(users))
        username = request.data["name"]
        password = request.data["password"]
        for user in users:
            print(user.email, user.password)
            if user.email == username:
                if user.password == password:
                    print("Creds are valid.")
                    access_token = str(random.randint(100000000000, 999999999999))
                    data = {'user_email': username, 'access_token': access_token, 'last_active': timezone.now()}
                    if ActiveUser.objects.filter(user_email=username).exists():
                        print("User is currently active")
                        update_user_last_active(username)
                        active_user = ActiveUser.objects.filter(user_email=username)[0]
                        return [str(active_user.access_token), username]
                    print("User is not currently active")
                    active_serializer = ActiveUserSerializer(data=data)
                    if active_serializer.is_valid():
                        active_serializer.save()
                    ActiveUser.delete_inactive_users()
                    return [str(access_token), username]

        return "Failed"
